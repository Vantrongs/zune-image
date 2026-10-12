//! Work reported as a decoder does it, so that its caller can stop it part way.
//!
//! A [`Progress`] is told the bytes a decoder reads (through a [`ProgressCursor`])
//! and the work of each row of MCUs it decodes, and stops it within a bounded amount
//! of work whatever the data.

use alloc::vec::Vec;

use zune_core::bytestream::{ZByteIoError, ZByteReaderTrait, ZSeekFrom};

use crate::errors::DecodeErrors;

/// Receives the work a decoder does and decides whether it goes on.
pub trait Progress {
    /// Counts `work` more bytes read or produced; true to stop. Once it returns true,
    /// it returns true for every later call, `work` 0 included.
    fn advance(&self, work: usize) -> bool;
}

/// Never stops.
impl Progress for () {
    fn advance(&self, _: usize) -> bool {
        false
    }
}

/// Whether `progress` stopped, without counting any work.
pub(crate) fn stopped(progress: &dyn Progress) -> bool {
    progress.advance(0)
}

/// Counts `work` on `progress`, failing with [`DecodeErrors::Stopped`] once it stops.
pub(crate) fn advance(progress: &dyn Progress, work: usize) -> Result<(), DecodeErrors> {
    if progress.advance(work) {
        Err(DecodeErrors::Stopped)
    } else {
        Ok(())
    }
}

/// The bytes made readable at a time beyond those a read needs.
const READ_PIECE: usize = 8192;

/// In-memory input that tells a [`Progress`] the bytes before reading them, a piece
/// at a time, and reads as ended once the progress stops, so that every loop over the
/// input ends. A read within the piece told costs one comparison, as a read of
/// [`ZCursor`](zune_core::bytestream::ZCursor) does; reads past the end count one
/// byte each.
pub struct ProgressCursor<'p, T> {
    data: T,
    position: usize,
    /// The bytes before this one were told to the progress.
    told: usize,
    progress: &'p dyn Progress,
    stopped: bool,
}

impl<'p, T: AsRef<[u8]>> ProgressCursor<'p, T> {
    /// `data` to read from the start, its reads told to `progress`.
    pub fn new(data: T, progress: &'p dyn Progress) -> Self {
        Self {
            data,
            position: 0,
            told: 0,
            progress,
            stopped: false,
        }
    }

    /// Tells the progress the bytes up to a piece past `end`, or one byte when there
    /// are none left; true when `position..end` can then be read.
    #[cold]
    fn tell(&mut self, end: usize) -> bool {
        if self.stopped {
            return false;
        }
        let len = self.data.as_ref().len();
        let told = len.min(end.saturating_add(READ_PIECE)).max(self.told);
        self.stopped = self.progress.advance((told - self.told).max(1));
        // Once stopped, every read comes here and finds nothing.
        self.told = if self.stopped { 0 } else { told };
        !self.stopped && end <= told
    }

    /// Copies the bytes from `position` into `buf` past the bytes told: fewer at the end
    /// of the data and none once the progress stopped.
    #[cold]
    fn peek_untold(&mut self, buf: &mut [u8]) -> usize {
        let end = self.position.saturating_add(buf.len());
        if !self.tell(end) && self.stopped {
            return 0;
        }
        let data = self.data.as_ref();
        let bytes = data.get(self.position..end.min(data.len())).unwrap_or(&[]);
        buf[..bytes.len()].copy_from_slice(bytes);
        bytes.len()
    }
}

impl<T: AsRef<[u8]>> ZByteReaderTrait for ProgressCursor<'_, T> {
    #[inline(always)]
    fn read_byte_no_error(&mut self) -> u8 {
        let byte = if self.position < self.told {
            self.data.as_ref()[self.position]
        } else {
            let mut byte = [0];
            self.peek_untold(&mut byte);
            byte[0]
        };
        self.position += 1;
        byte
    }

    #[inline(always)]
    fn read_exact_bytes(&mut self, buf: &mut [u8]) -> Result<(), ZByteIoError> {
        self.peek_exact_bytes(buf)?;
        self.position += buf.len();
        Ok(())
    }

    #[inline(always)]
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<usize, ZByteIoError> {
        let read = self.peek_bytes(buf)?;
        self.position += read;
        Ok(read)
    }

    #[inline(always)]
    fn peek_bytes(&mut self, buf: &mut [u8]) -> Result<usize, ZByteIoError> {
        let end = self.position + buf.len();
        if end <= self.told {
            buf.copy_from_slice(&self.data.as_ref()[self.position..end]);
            return Ok(buf.len());
        }
        Ok(self.peek_untold(buf))
    }

    #[inline(always)]
    fn peek_exact_bytes(&mut self, buf: &mut [u8]) -> Result<(), ZByteIoError> {
        let read = self.peek_bytes(buf)?;
        if read != buf.len() {
            return Err(ZByteIoError::NotEnoughBytes(read, buf.len()));
        }
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn z_seek(&mut self, from: ZSeekFrom) -> Result<u64, ZByteIoError> {
        let (base, offset) = match from {
            ZSeekFrom::Start(n) => {
                self.position = n as usize;
                return Ok(n);
            }
            ZSeekFrom::End(n) => (self.data.as_ref().len(), n as isize),
            ZSeekFrom::Current(n) => (self.position, n as isize),
        };
        self.position = base
            .checked_add_signed(offset)
            .ok_or(ZByteIoError::SeekError("Negative seek"))?;
        Ok(self.position as u64)
    }

    fn is_eof(&mut self) -> Result<bool, ZByteIoError> {
        Ok(self.stopped || self.position >= self.data.as_ref().len())
    }

    fn z_position(&mut self) -> Result<u64, ZByteIoError> {
        Ok(self.position as u64)
    }

    fn read_remaining(&mut self, sink: &mut Vec<u8>) -> Result<usize, ZByteIoError> {
        let mut piece = [0; READ_PIECE];
        let mut total = 0;
        loop {
            let read = self.read_bytes(&mut piece)?;
            if read == 0 {
                return Ok(total);
            }
            sink.extend_from_slice(&piece[..read]);
            total += read;
        }
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use zune_core::bytestream::{ZByteReaderTrait, ZCursor, ZSeekFrom};

    use super::{Progress, ProgressCursor, READ_PIECE};

    struct StopAfter {
        work: Cell<usize>,
        limit: usize,
    }

    impl Progress for StopAfter {
        fn advance(&self, work: usize) -> bool {
            self.work.set(self.work.get() + work);
            self.work.get() >= self.limit
        }
    }

    /// Reads are told a piece ahead, and once the progress stops the input reads as
    /// ended: the read the progress stopped at finds nothing.
    #[test]
    fn reads_end_once_the_progress_stops() {
        let bytes = alloc::vec![0xff_u8; 4 * READ_PIECE];
        let progress = StopAfter {
            work: Cell::new(0),
            limit: 2 * READ_PIECE,
        };
        let mut reader = ProgressCursor::new(&bytes, &progress);
        let mut read = 0;
        while reader.read_byte_no_error() == 0xff {
            read += 1;
        }
        assert_eq!(read, READ_PIECE + 1);
        assert!(reader.is_eof().unwrap());
        let mut buf = [0; 4];
        assert_eq!(reader.read_bytes(&mut buf).unwrap(), 0);
        assert!(reader.read_exact_bytes(&mut buf).is_err());
    }

    /// Reads past the end are told one byte each, so that a loop of them ends too.
    #[test]
    fn reads_past_the_end_are_told() {
        let bytes = [7_u8; 3];
        let progress = StopAfter {
            work: Cell::new(0),
            limit: 10,
        };
        let mut reader = ProgressCursor::new(&bytes, &progress);
        let mut reads = 0;
        while !progress.advance(0) {
            reader.read_byte_no_error();
            reads += 1;
        }
        assert_eq!(reads, 3 + 7);
    }

    /// Without a stop, every read, peek, seek and failed read is as `ZCursor`'s.
    #[test]
    fn reads_are_as_a_cursor_s() {
        let bytes: alloc::vec::Vec<u8> = (0..=u8::MAX).cycle().take(3 * READ_PIECE + 5).collect();
        let mut a = ZCursor::new(&bytes);
        let mut b = ProgressCursor::new(&bytes, &());
        let at = 3 * READ_PIECE;
        for step in 0..20 {
            match step % 5 {
                0 => assert_eq!(a.read_byte_no_error(), b.read_byte_no_error()),
                1 => {
                    let (mut x, mut y) = ([0; 9], [0; 9]);
                    assert_eq!(
                        a.read_exact_bytes(&mut x).is_ok(),
                        b.read_exact_bytes(&mut y).is_ok()
                    );
                    assert_eq!(x, y);
                }
                2 => {
                    let (mut x, mut y) = ([0; 7], [0; 7]);
                    assert_eq!(a.peek_bytes(&mut x).unwrap(), b.peek_bytes(&mut y).unwrap());
                    assert_eq!(x, y);
                    assert_eq!(
                        a.peek_exact_bytes(&mut x).is_ok(),
                        b.peek_exact_bytes(&mut y).is_ok()
                    );
                }
                3 => {
                    let (mut x, mut y) = ([0; 5000], [0; 5000]);
                    assert_eq!(a.read_bytes(&mut x).unwrap(), b.read_bytes(&mut y).unwrap());
                    assert_eq!(x, y);
                }
                _ => {
                    let to = [
                        ZSeekFrom::Start(at as u64),
                        ZSeekFrom::Current(-11),
                        ZSeekFrom::End(-2),
                    ];
                    let to = to[step / 5 % 3];
                    assert_eq!(a.z_seek(to).unwrap(), b.z_seek(to).unwrap());
                }
            }
            assert_eq!(a.z_position().unwrap(), b.z_position().unwrap());
            assert_eq!(a.is_eof().unwrap(), b.is_eof().unwrap());
        }
    }
}
