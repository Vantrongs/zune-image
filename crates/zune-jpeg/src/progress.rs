//! Work reported as a decoder does it, so that its caller can stop it part way.
//!
//! A [`Progress`] is told the bytes a decoder reads (through a [`ProgressReader`])
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

/// The bytes read between reports to the progress.
const READ_PIECE: usize = 8192;

/// An input that reports the bytes read from it to a [`Progress`], and reads as
/// ended once the progress stops, so that every loop over the input ends.
pub struct ProgressReader<'p, R> {
    inner: R,
    progress: &'p dyn Progress,
    unreported: usize,
    stopped: bool,
}

impl<'p, R> ProgressReader<'p, R> {
    /// `inner`, its reads reported to `progress`.
    pub fn new(inner: R, progress: &'p dyn Progress) -> Self {
        Self {
            inner,
            progress,
            unreported: 0,
            stopped: false,
        }
    }

    /// Counts a read of up to `bytes`, at least one; false once the progress stopped,
    /// for the read to find nothing.
    #[inline(always)]
    fn read(&mut self, bytes: usize) -> bool {
        self.unreported += bytes.max(1);
        self.unreported < READ_PIECE || self.report()
    }

    /// Reports the reads counted; once stopped, every later read comes here.
    #[cold]
    fn report(&mut self) -> bool {
        if !self.stopped {
            self.stopped = self.progress.advance(core::mem::take(&mut self.unreported));
        }
        if self.stopped {
            self.unreported = READ_PIECE;
        }
        !self.stopped
    }
}

impl<R: ZByteReaderTrait> ZByteReaderTrait for ProgressReader<'_, R> {
    #[inline(always)]
    fn read_byte_no_error(&mut self) -> u8 {
        if !self.read(1) {
            return 0;
        }
        self.inner.read_byte_no_error()
    }

    #[inline(always)]
    fn read_exact_bytes(&mut self, buf: &mut [u8]) -> Result<(), ZByteIoError> {
        if !self.read(buf.len()) {
            return Err(ZByteIoError::NotEnoughBytes(0, buf.len()));
        }
        self.inner.read_exact_bytes(buf)
    }

    #[inline(always)]
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<usize, ZByteIoError> {
        if !self.read(buf.len()) {
            return Ok(0);
        }
        self.inner.read_bytes(buf)
    }

    #[inline(always)]
    fn peek_bytes(&mut self, buf: &mut [u8]) -> Result<usize, ZByteIoError> {
        if !self.read(buf.len()) {
            return Ok(0);
        }
        self.inner.peek_bytes(buf)
    }

    #[inline(always)]
    fn peek_exact_bytes(&mut self, buf: &mut [u8]) -> Result<(), ZByteIoError> {
        if !self.read(buf.len()) {
            return Err(ZByteIoError::NotEnoughBytes(0, buf.len()));
        }
        self.inner.peek_exact_bytes(buf)
    }

    fn z_seek(&mut self, from: ZSeekFrom) -> Result<u64, ZByteIoError> {
        self.inner.z_seek(from)
    }

    fn is_eof(&mut self) -> Result<bool, ZByteIoError> {
        if self.stopped {
            return Ok(true);
        }
        self.inner.is_eof()
    }

    fn z_position(&mut self) -> Result<u64, ZByteIoError> {
        self.inner.z_position()
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

    use zune_core::bytestream::{ZByteReaderTrait, ZCursor};

    use super::{Progress, ProgressReader, READ_PIECE};

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

    /// Reads are reported a piece at a time, and once the progress stops the input
    /// reads as ended: the read that the progress stopped at finds nothing.
    #[test]
    fn reads_end_once_the_progress_stops() {
        let bytes = alloc::vec![0xff_u8; 4 * READ_PIECE];
        let progress = StopAfter {
            work: Cell::new(0),
            limit: 2 * READ_PIECE,
        };
        let mut reader = ProgressReader::new(ZCursor::new(&bytes), &progress);
        let mut read = 0;
        while reader.read_byte_no_error() == 0xff {
            read += 1;
        }
        assert_eq!(read, 2 * READ_PIECE - 1);
        assert!(reader.is_eof().unwrap());
        let mut buf = [0; 4];
        assert_eq!(reader.read_bytes(&mut buf).unwrap(), 0);
        assert!(reader.read_exact_bytes(&mut buf).is_err());
    }
}
