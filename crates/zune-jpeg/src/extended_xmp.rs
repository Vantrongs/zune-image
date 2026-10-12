/*
 * Copyright (c) 2023.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Extended XMP segments, reassembled as they arrive.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use zune_core::log::error;

#[derive(Default, Clone, Eq, PartialEq, Debug)]
pub(crate) struct ExtendedXmpSegment {
    pub(crate) offset:     u32,
    pub(crate) total_size: u32,
    pub(crate) guid:       Vec<u8>,
    pub(crate) data:       Vec<u8>
}

/// The Extended XMP segments read so far, of one GUID and total size.
///
/// In order of offset, ties in order of arrival, the segments from offset 0 that
/// each start where the ones before end are joined; the rest are held until they
/// can be. The data is complete once every segment is joined and it ends at the
/// total size. Each segment is held and joined at most once, so the work grows with
/// the number of segments times its logarithm.
#[derive(Default)]
pub(crate) struct ExtendedXmp {
    /// The GUID and total size of the segments read; `None` before the first.
    id:       Option<(Vec<u8>, u32)>,
    joined:   Vec<u8>,
    /// The end of the joined data.
    end:      u64,
    /// By offset and arrival. Once the first one starts before `end`, it can never
    /// be joined, nor can the segments be complete.
    held:     BTreeMap<(u32, usize), Vec<u8>>,
    arrivals: usize
}

impl ExtendedXmp {
    /// Adds `segment`; returns the GUID and data once the segments are complete, and
    /// starts over. A segment of another GUID or total size than those read, or
    /// joined data past the total size, drops the segments.
    pub(crate) fn add(&mut self, segment: ExtendedXmpSegment) -> Option<(Vec<u8>, Vec<u8>)> {
        let ExtendedXmpSegment {
            offset,
            total_size,
            guid,
            data
        } = segment;
        match &self.id {
            None => self.id = Some((guid, total_size)),
            Some((read, size)) if *read != guid || *size != total_size => {
                error!("Inconsistent Extended XMP segments");
                *self = Self::default();
                return None;
            }
            Some(_) => {}
        }
        if u64::from(offset) != self.end || self.first_held().is_some_and(|first| first <= offset) {
            self.hold(offset, data);
            return None;
        }
        self.join(&data);
        while let Some(entry) = self.held.first_entry() {
            if u64::from(entry.key().0) != self.end {
                break;
            }
            let data = entry.remove();
            self.join(&data);
        }
        if self.held.is_empty() && self.end == u64::from(total_size) {
            let (guid, _) = self.id.take()?;
            let data = core::mem::take(&mut self.joined);
            *self = Self::default();
            return Some((guid, data));
        }
        if self.end > u64::from(total_size) {
            error!("Extended XMP overflow");
            *self = Self::default();
        }
        None
    }

    fn first_held(&self) -> Option<u32> {
        self.held.keys().next().map(|&(offset, _)| offset)
    }

    fn hold(&mut self, offset: u32, data: Vec<u8>) {
        #[cfg(test)]
        tests::STEPS.with(|steps| steps.set(steps.get() + 1));
        self.held.insert((offset, self.arrivals), data);
        self.arrivals += 1;
    }

    fn join(&mut self, data: &[u8]) {
        #[cfg(test)]
        tests::STEPS.with(|steps| steps.set(steps.get() + 1));
        self.joined.extend_from_slice(data);
        self.end += data.len() as u64;
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use alloc::vec;
    use core::cell::Cell;

    use super::*;

    std::thread_local! {
        /// The segments held or joined on this thread.
        pub(super) static STEPS: Cell<usize> = const { Cell::new(0) };
    }

    /// How the decoder reassembled the segments before, after each one arrived: sort
    /// all of them by offset and join them if they are complete.
    #[derive(Default)]
    struct Reference(Vec<ExtendedXmpSegment>);

    impl Reference {
        fn add(&mut self, segment: ExtendedXmpSegment) -> Option<(Vec<u8>, Vec<u8>)> {
            let segments = &mut self.0;
            segments.push(segment);
            segments.sort_by_key(|s| s.offset);
            let guid = segments[0].guid.clone();
            let total_size = segments[0].total_size;
            if segments
                .iter()
                .any(|s| s.guid != guid || s.total_size != total_size)
            {
                segments.clear();
                return None;
            }
            let mut rolling = 0;
            let mut complete = true;
            for segment in segments.iter() {
                if segment.offset != rolling {
                    complete = false;
                    break;
                }
                rolling += segment.data.len() as u32;
            }
            if complete && rolling == total_size {
                let data = segments.iter().flat_map(|s| s.data.clone()).collect();
                segments.clear();
                return Some((guid, data));
            }
            if rolling > total_size {
                segments.clear();
            }
            None
        }
    }

    /// A small deterministic generator of pseudo-random numbers.
    struct Numbers(u64);

    impl Numbers {
        fn below(&mut self, n: u32) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % u64::from(n)) as u32
        }

        /// `usual` three times in four, otherwise any number below `any`.
        fn mostly(&mut self, usual: u32, any: u32) -> u32 {
            match self.below(4) {
                0 => self.below(any),
                _ => usual
            }
        }
    }

    /// Segments in any order, with gaps, overlaps, repeats, empty data, other GUIDs and
    /// total sizes, complete the same way as sorting them all after each arrival does.
    #[test]
    fn segments_complete_as_sorting_them_all_does() {
        let mut numbers = Numbers(0x9e37_79b9_7f4a_7c15);
        let mut completed = 0;
        for _ in 0..20_000 {
            let (mut new, mut reference) = (ExtendedXmp::default(), Reference::default());
            for _ in 0..=numbers.below(12) {
                // Mostly pieces of two bytes that tile six.
                let usual = 2 * numbers.below(3);
                let offset = numbers.mostly(usual, 10);
                let len = numbers.mostly(2, 4);
                let total_size = numbers.mostly(6, 9);
                let guid = vec![b'a' + numbers.mostly(0, 2) as u8];
                let segment = ExtendedXmpSegment {
                    offset,
                    total_size,
                    guid,
                    data: (0..len).map(|_| numbers.below(256) as u8).collect()
                };
                let expected = reference.add(segment.clone());
                completed += usize::from(expected.is_some());
                assert_eq!(new.add(segment), expected);
            }
        }
        assert!(completed > 1000, "{completed} completed");
    }

    /// Each segment is held and joined at most once, whatever the order of arrival,
    /// also when segments of offset 0 and no data never complete.
    #[test]
    fn the_work_grows_with_the_segments() {
        let count = 4096;
        let segment = |offset: u32, len: usize| ExtendedXmpSegment {
            offset,
            total_size: if len == 0 { 1 } else { count * 2 },
            guid: vec![7; 32],
            data: vec![1; len]
        };
        for order in [
            (0..count).map(|_| segment(0, 0)).collect::<Vec<_>>(),
            (0..count).rev().map(|i| segment(i * 2, 2)).collect(),
            (0..count).map(|i| segment(i * 2, 2)).collect()
        ] {
            STEPS.with(|steps| steps.set(0));
            let mut xmp = ExtendedXmp::default();
            let completed = order.into_iter().filter_map(|s| xmp.add(s)).count();
            assert!(completed <= 1);
            assert!(STEPS.with(Cell::get) <= 2 * count as usize);
        }
    }
}
