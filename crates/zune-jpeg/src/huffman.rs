/*
 * Copyright (c) 2023.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! This file contains the struct `HuffmanTable` that stores Huffman tables
//! needed during `BitStream` decoding, and `HuffmanTables`, a decoder's set of them.
#![allow(clippy::similar_names, clippy::module_name_repetitions)]

use alloc::boxed::Box;
use alloc::format;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use crate::decoder::MAX_COMPONENTS;
use crate::errors::DecodeErrors;

/// Determines how many bits of lookahead we have for our bitstream decoder.
pub const HUFF_LOOKAHEAD: u8 = 9;

/// A struct which contains necessary tables for decoding a JPEG
/// huffman encoded bitstream
pub struct HuffmanTable {
    // element `[0]` of each array is unused
    /// largest code of length k
    pub(crate) maxcode: [i32; 18],
    /// offset for codes of length k
    /// Answers the question, where do code-lengths of length k end
    /// Element 0 is unused
    pub(crate) offset:  [i32; 18],
    /// lookup table for fast decoding
    ///
    /// top  bits above HUFF_LOOKAHEAD contain the code length.
    ///
    /// Lower (8) bits contain the symbol in order of increasing code length.
    pub(crate) lookup:  [i32; 1 << HUFF_LOOKAHEAD],

    /// A table which can be used to decode small AC coefficients and
    /// do an equivalent of receive_extend
    pub(crate) ac_lookup: Option<[i16; 1 << HUFF_LOOKAHEAD]>,

    /// Directly represent contents of a JPEG DHT marker
    ///
    /// \# number of symbols with codes of length `k` bits
    // bits[0] is unused
    /// Symbols in order of increasing code length
    pub(crate) values: [u8; 256]
}

impl HuffmanTable {
    pub(crate) fn validate_dc_symbol_limit(&self, limit: u8) -> Result<(), DecodeErrors> {
        if self.values.iter().any(|&symbol| symbol > limit) {
            return Err(DecodeErrors::HuffmanDecode("Bad Huffman Table".to_string()));
        }
        Ok(())
    }

    /// Builds the table in `slot`, where decoding will read it, rather than on the
    /// stack to be copied there; leaves `slot` empty if the table is invalid.
    fn build_in(
        slot: &mut Option<InstalledTable>, codes: &[u8; 17], values: &[u8; 256], is_dc: bool,
        is_progressive: bool, dc_symbol_limit: u8
    ) -> Result<(), DecodeErrors> {
        let too_long_code = (i32::from(HUFF_LOOKAHEAD) + 1) << HUFF_LOOKAHEAD;
        let InstalledTable(table) = slot.insert(InstalledTable(HuffmanTable {
            maxcode: [0; 18],
            offset: [0; 18],
            lookup: [too_long_code; 1 << HUFF_LOOKAHEAD],
            values: *values,
            ac_lookup: None
        }));
        let result = table.make_derived_table(is_dc, is_progressive, codes, dc_symbol_limit);
        if result.is_err() {
            *slot = None;
        }
        result
    }

    /// Compute derived values for a Huffman table
    ///
    /// This routine performs some validation checks on the table
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        clippy::needless_range_loop,
        clippy::explicit_counter_loop,
    )]
    fn make_derived_table(
        &mut self, is_dc: bool, _is_progressive: bool, bits: &[u8; 17], dc_symbol_limit: u8
    ) -> Result<(), DecodeErrors> {
        // build a list of code size
        let mut huff_size = [0; 257];
        // Huffman code lengths
        let mut huff_code: [u32; 257] = [0; 257];
        // figure C.1 make table of Huffman code length for each symbol
        let mut p = 0;

        for l in 1..=16 {
            let mut i = i32::from(bits[l]);
            // table overrun is checked before ,so we dont need to check
            while i != 0 {
                huff_size[p] = l as u8;
                p += 1;
                i -= 1;
            }
        }

        huff_size[p] = 0;

        let num_symbols = p;
        // Generate the codes themselves
        // We also validate that the counts represent a legal Huffman code tree
        let mut code = 0;
        let mut si = i32::from(huff_size[0]);

        p = 0;

        while huff_size[p] != 0 {
            while i32::from(huff_size[p]) == si {
                huff_code[p] = code;
                code += 1;
                p += 1;
            }
            // maximum code of length si, pre-shifted by 16-k bits
            self.maxcode[si as usize] = (code << (16 - si)) as i32;
            // code is now 1 more than the last code used for code-length si; but
            // it must still fit in si bits, since no code is allowed to be all ones.
            if (code as i32) >= (1 << si) {
                return Err(DecodeErrors::HuffmanDecode("Bad Huffman Table".to_string()));
            }

            code <<= 1;
            si += 1;
        }

        // Figure F.15 generate decoding tables for bit-sequential decoding
        p = 0;

        for l in 0..=16 {
            if bits[l] == 0 {
                // -1 if no codes of this length
                self.maxcode[l] = -1;
            } else {
                // offset[l]=codes[index of 1st symbol of code length l
                // minus minimum code of length l]
                self.offset[l] = (p as i32) - (huff_code[p]) as i32;
                p += usize::from(bits[l]);
            }
        }

        self.offset[17] = 0;
        // we ensure that decode terminates
        self.maxcode[17] = 0x000F_FFFF;

        /*
         * Compute lookahead tables to speed up decoding.
         * First we set all the table entries to 0(left justified), indicating "too long";
         * (Note too long was set during initialization)
         * then we iterate through the Huffman codes that are short enough and
         * fill in all the entries that correspond to bit sequences starting
         * with that code.
         */

        p = 0;

        for l in 1..=HUFF_LOOKAHEAD {
            let entries = 1 << (HUFF_LOOKAHEAD - l);
            for _ in 1..=i32::from(bits[usize::from(l)]) {
                // l -> Current code length,
                // p => Its index in self.code and self.values
                // Every bit sequence starting with this left-justified code decodes to it.
                let look_bits = (huff_code[p] as usize) << (HUFF_LOOKAHEAD - l);
                self.lookup[look_bits..look_bits + entries]
                    .fill((i32::from(l) << HUFF_LOOKAHEAD) | i32::from(self.values[p]));

                p += 1;
            }
        }
        // build an ac table that does an equivalent of decode and receive_extend
        if !is_dc {
            // Built per code rather than per entry: every entry a code of at most
            // HUFF_LOOKAHEAD bits starts holds what that code and the bits after it
            // decode to, in code order, so a later code wins as it would per entry.
            // (The entry-by-entry build skipped the symbol at index 255. It never gets
            // a value here, which takes a code of at most 8 bits with its magnitude
            // bits, and no table has 256 codes of at most 8 bits.)
            let fast_ac = self.ac_lookup.insert([0; 1 << HUFF_LOOKAHEAD]);
            for i in 0..num_symbols {
                // get code size for an item
                let s = huff_size[i];

                // Codes come in order of length.
                if s > HUFF_LOOKAHEAD {
                    break;
                }
                let c = (huff_code[i] << (HUFF_LOOKAHEAD - s)) as usize;
                let entries = &mut fast_ac[c..c + (1 << (HUFF_LOOKAHEAD - s))];

                // get symbol value from AC table
                let rs = self.values[i];
                // shift by 4 to get run length
                let run = i16::from((rs >> 4) & 15);
                // get magnitude bits stored at the lower 3 bits
                let mag_bits = i16::from(rs & 15);
                // length of the bit we've read
                let len = i16::from(s);

                if mag_bits == 0 || (len + mag_bits) > i16::from(HUFF_LOOKAHEAD) {
                    entries.fill(0);
                    continue;
                }
                // The magnitude code is the top `mag_bits` of the bits after the code.
                let shift = i16::from(HUFF_LOOKAHEAD) - len - mag_bits;
                let m = 1 << (mag_bits - 1);
                for (j, entry) in entries.iter_mut().enumerate() {
                    // magnitude code followed by receive_extend code
                    let mut k = (j >> shift) as i16;

                    if k < m {
                        k += (!0_i16 << mag_bits) + 1;
                    }

                    // if result is small enough fit into fast ac table
                    *entry = if (-128..=127).contains(&k) {
                        (k << 8) + (run << 4) + (len + mag_bits)
                    } else {
                        0
                    };
                }
            }
        }

        // Validate symbols as being reasonable
        // For AC tables, we make no check, but accept all byte values 0..255
        // For DC tables, we require symbols to be in range 0..15
        if is_dc {
            for i in 0..num_symbols {
                let sym = self.values[i];

                if sym > dc_symbol_limit {
                    return Err(DecodeErrors::HuffmanDecode("Bad Huffman Table".to_string()));
                }
            }
        }

        Ok(())
    }
}

/// The class of a Huffman table: DC or AC.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TableClass {
    Dc = 0,
    Ac = 1
}

/// A Huffman table in a [`HuffmanTables`] slot. Decoding only reads it: it has no
/// public constructor and no `Clone`, so only [`HuffmanTables`] puts one in a slot.
pub(crate) struct InstalledTable(HuffmanTable);

impl Deref for InstalledTable {
    type Target = HuffmanTable;

    #[inline(always)]
    fn deref(&self) -> &HuffmanTable {
        &self.0
    }
}

impl InstalledTable {
    fn copy(&self) -> Self {
        let t = &self.0;
        InstalledTable(HuffmanTable {
            maxcode:   t.maxcode,
            offset:    t.offset,
            lookup:    t.lookup,
            ac_lookup: t.ac_lookup,
            values:    t.values
        })
    }
}

type Slots = [[Option<InstalledTable>; MAX_COMPONENTS]; 2];

fn copy_slots(slots: &Slots) -> Slots {
    slots.each_ref().map(|class| {
        class
            .each_ref()
            .map(|slot| slot.as_ref().map(InstalledTable::copy))
    })
}

/// A decoder's DC and AC Huffman tables, four of each, kept in place.
///
/// A scan snapshot (taken at the first SOS, and at scan boundaries when decoding
/// can resume) must give back the tables in effect when it was taken, while a DHT
/// between scans may replace them. Copying the eight slots (about 28 KiB) into each
/// snapshot, or giving each table its own allocation for snapshots to share, costs
/// more than decoding a small image, and the tables rarely change once a snapshot
/// exists. So a snapshot holds a token of the tables' current version, and a change
/// first copies the tables aside if a snapshot still holds that version's token.
/// Every change goes through this type, whose fields no other module can reach.
pub(crate) struct HuffmanTables {
    slots:   Slots,
    /// The current version's token, made when a snapshot first needs it.
    version: Option<Arc<()>>,
    /// Earlier versions' tables, kept while a snapshot holds their token.
    kept:    Vec<(Arc<()>, Box<Slots>)>
}

/// The tables being defined by one [`HuffmanTables::define`].
///
/// A table for an empty slot is built in that slot, which a failure empties again;
/// one replacing a defined table is built aside and moved in once all succeed.
pub(crate) struct Definitions<'a> {
    tables:   &'a mut HuffmanTables,
    /// Whether a table was defined, which changes the tables.
    began:    bool,
    /// The slots that were empty and have a table built in place.
    in_place: [[bool; MAX_COMPONENTS]; 2],
    staged:   Slots
}

impl Definitions<'_> {
    pub(crate) fn is_defined(&self, class: TableClass, index: usize) -> bool {
        self.tables.slots[class as usize][index].is_some()
    }

    /// Builds a table for a slot; a DC table's symbols may be at most `dc_symbol_limit`.
    pub(crate) fn define(
        &mut self, class: TableClass, index: usize, codes: &[u8; 17], values: &[u8; 256],
        is_progressive: bool, dc_symbol_limit: u8
    ) -> Result<(), DecodeErrors> {
        if !self.began {
            self.began = true;
            self.tables.before_change();
        }
        let (c, i) = (class as usize, index);
        let slot = if self.tables.slots[c][i].is_none() || self.in_place[c][i] {
            self.in_place[c][i] = true;
            &mut self.tables.slots[c][i]
        } else {
            &mut self.staged[c][i]
        };
        HuffmanTable::build_in(
            slot,
            codes,
            values,
            class == TableClass::Dc,
            is_progressive,
            dc_symbol_limit
        )
    }
}

/// The Huffman tables a [`HuffmanTables`] had when [`HuffmanTables::snapshot`] was called.
#[derive(Clone)]
pub(crate) struct HuffmanSnapshot(Arc<()>);

impl HuffmanTables {
    pub(crate) const fn new() -> Self {
        HuffmanTables {
            slots:   [[None, None, None, None], [None, None, None, None]],
            version: None,
            kept:    Vec::new()
        }
    }

    /// The table in a slot, or the error for a slot index past the end or an empty slot.
    #[inline(always)]
    pub(crate) fn get_mut(
        &mut self, class: TableClass, index: usize
    ) -> Result<&mut InstalledTable, DecodeErrors> {
        Self::slot(&mut self.slots[class as usize], class, index)
    }

    /// The DC and the AC table a component decodes with.
    #[inline(always)]
    pub(crate) fn get_pair_mut(
        &mut self, dc_index: usize, ac_index: usize
    ) -> Result<(&mut InstalledTable, &mut InstalledTable), DecodeErrors> {
        let [dc, ac] = &mut self.slots;
        Ok((
            Self::slot(dc, TableClass::Dc, dc_index)?,
            Self::slot(ac, TableClass::Ac, ac_index)?
        ))
    }

    #[inline(always)]
    fn slot(
        slots: &mut [Option<InstalledTable>; MAX_COMPONENTS], class: TableClass, index: usize
    ) -> Result<&mut InstalledTable, DecodeErrors> {
        let name = if class == TableClass::Dc { "DC" } else { "AC" };
        slots
            .get_mut(index)
            .ok_or_else(|| {
                DecodeErrors::Format(format!("No huffman table for {name} component: {index}"))
            })?
            .as_mut()
            .ok_or_else(|| {
                DecodeErrors::Format(format!(
                    "{name} Huffman table at index {index} not initialized"
                ))
            })
    }

    /// The DC tables that are defined.
    pub(crate) fn dc_tables(&self) -> impl Iterator<Item = &HuffmanTable> {
        self.slots[TableClass::Dc as usize].iter().flatten().map(|table| &table.0)
    }

    /// Defines the tables `define` gives: all of them if it succeeds, none if it
    /// fails, as a DHT whose body turns out malformed must leave the tables alone.
    pub(crate) fn define<R>(
        &mut self, define: impl FnOnce(&mut Definitions<'_>) -> Result<R, DecodeErrors>
    ) -> Result<R, DecodeErrors> {
        let mut definitions = Definitions {
            tables:   self,
            began:    false,
            in_place: [[false; MAX_COMPONENTS]; 2],
            staged:   [[None, None, None, None], [None, None, None, None]]
        };
        let result = define(&mut definitions);
        let Definitions { tables, in_place, staged, .. } = &mut definitions;
        for ((slots, in_place), staged) in tables.slots.iter_mut().zip(in_place).zip(staged) {
            for ((slot, in_place), staged) in slots.iter_mut().zip(in_place).zip(staged) {
                if result.is_err() {
                    if *in_place {
                        *slot = None;
                    }
                } else if staged.is_some() {
                    *slot = staged.take();
                }
            }
        }
        result
    }

    pub(crate) fn snapshot(&mut self) -> HuffmanSnapshot {
        HuffmanSnapshot(Arc::clone(self.version.get_or_insert_with(|| Arc::new(()))))
    }

    /// Gives the tables back the contents they had when `snapshot` was taken.
    pub(crate) fn restore(&mut self, snapshot: &HuffmanSnapshot) -> Result<(), DecodeErrors> {
        if self.version.as_ref().is_some_and(|version| Arc::ptr_eq(version, &snapshot.0)) {
            return Ok(());
        }
        self.before_change();
        // A change kept these tables, as the snapshot held their token then and still does.
        let (_, kept) = self
            .kept
            .iter()
            .find(|(version, _)| Arc::ptr_eq(version, &snapshot.0))
            .ok_or(DecodeErrors::FormatStatic(
                "Huffman tables of a scan snapshot were not kept"
            ))?;
        self.slots = copy_slots(kept);
        self.version = Some(Arc::clone(&snapshot.0));
        Ok(())
    }

    /// Keeps the current tables if a snapshot holds their version, which then ends,
    /// and drops the kept tables no snapshot holds any more.
    fn before_change(&mut self) {
        if let Some(version) = self.version.take() {
            let kept = self.kept.iter().any(|(v, _)| Arc::ptr_eq(v, &version));
            if Arc::strong_count(&version) > 1 && !kept {
                self.kept.push((version, Box::new(copy_slots(&self.slots))));
            }
        }
        self.kept.retain(|(version, _)| Arc::strong_count(version) > 1);
    }
}

#[cfg(test)]
mod tests {
    use super::{HuffmanTable, HuffmanTables, TableClass, HUFF_LOOKAHEAD};

    fn install(tables: &mut HuffmanTables, class: TableClass, index: usize, symbol: u8) {
        let mut bits = [0; 17];
        bits[1] = 1;
        let mut values = [0; 256];
        values[0] = symbol;
        tables.define(|tables| tables.define(class, index, &bits, &values, false, 15)).unwrap();
    }

    fn symbol(tables: &mut HuffmanTables, class: TableClass, index: usize) -> u8 {
        tables.get_mut(class, index).unwrap().values[0]
    }

    #[test]
    fn a_failed_definition_leaves_the_tables_alone() {
        let mut tables = HuffmanTables::new();
        install(&mut tables, TableClass::Dc, 0, 1);
        let snapshot = tables.snapshot();
        let mut bits = [0; 17];
        bits[1] = 1;
        // Two codes of one bit leave no room for a code that is not all ones.
        let mut bad_bits = [0; 17];
        bad_bits[1] = 2;
        let values = [7; 256];
        let result = tables.define(|tables| {
            // Replaces a defined table, built aside.
            tables.define(TableClass::Dc, 0, &bits, &values, false, 15)?;
            // Fills an empty slot, built in place, twice.
            tables.define(TableClass::Ac, 3, &bits, &values, false, 15)?;
            tables.define(TableClass::Ac, 3, &bits, &values, false, 15)?;
            tables.define(TableClass::Ac, 0, &bad_bits, &values, false, 15)
        });
        assert!(result.is_err());
        assert_eq!(symbol(&mut tables, TableClass::Dc, 0), 1);
        assert!(tables.get_mut(TableClass::Ac, 3).is_err());
        assert!(tables.get_mut(TableClass::Ac, 0).is_err());
        tables.restore(&snapshot).unwrap();
        assert_eq!(symbol(&mut tables, TableClass::Dc, 0), 1);

        tables
            .define(|tables| {
                tables.define(TableClass::Dc, 0, &bits, &values, false, 15)?;
                tables.define(TableClass::Ac, 3, &bits, &values, false, 15)
            })
            .unwrap();
        assert_eq!(
            (symbol(&mut tables, TableClass::Dc, 0), symbol(&mut tables, TableClass::Ac, 3)),
            (7, 7)
        );
        tables.restore(&snapshot).unwrap();
        assert_eq!(symbol(&mut tables, TableClass::Dc, 0), 1);
        assert!(tables.get_mut(TableClass::Ac, 3).is_err());
    }

    #[test]
    fn a_snapshot_gives_back_the_tables_it_was_taken_with() {
        let mut tables = HuffmanTables::new();
        install(&mut tables, TableClass::Dc, 0, 1);
        install(&mut tables, TableClass::Ac, 1, 2);
        let first = tables.snapshot();
        // A snapshot of unchanged tables shares their version.
        let same = tables.snapshot();
        tables.restore(&same).unwrap();
        assert!(tables.kept.is_empty());

        install(&mut tables, TableClass::Dc, 0, 3);
        assert_eq!(tables.kept.len(), 1);
        let second = tables.snapshot();
        install(&mut tables, TableClass::Dc, 0, 4);
        install(&mut tables, TableClass::Ac, 2, 5);
        assert_eq!(tables.kept.len(), 2);

        tables.restore(&first).unwrap();
        assert_eq!(
            (symbol(&mut tables, TableClass::Dc, 0), symbol(&mut tables, TableClass::Ac, 1)),
            (1, 2)
        );
        assert!(tables.get_mut(TableClass::Ac, 2).is_err());
        tables.restore(&second).unwrap();
        assert_eq!(symbol(&mut tables, TableClass::Dc, 0), 3);
        tables.restore(&first).unwrap();
        tables.restore(&same).unwrap();
        assert_eq!(symbol(&mut tables, TableClass::Dc, 0), 1);

        // Kept tables go once no snapshot holds them.
        drop((first, same, second));
        install(&mut tables, TableClass::Dc, 1, 6);
        assert!(tables.kept.is_empty());
        assert_eq!(
            (symbol(&mut tables, TableClass::Dc, 0), symbol(&mut tables, TableClass::Dc, 1)),
            (1, 6)
        );
    }

    /// The lookahead tables as zune-jpeg 0.5.16 built them, entry by entry.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn reference_lookahead(
        bits: &[u8; 17], values: &[u8; 256], is_dc: bool
    ) -> ([i32; 1 << HUFF_LOOKAHEAD], Option<[i16; 1 << HUFF_LOOKAHEAD]>) {
        let mut huff_size = [0_u8; 257];
        let mut huff_code = [0_u32; 257];
        let mut p = 0;
        for l in 1..=16 {
            for _ in 0..bits[l] {
                huff_size[p] = l as u8;
                p += 1;
            }
        }
        let num_symbols = p;
        let (mut code, mut si, mut p) = (0_u32, huff_size[0], 0);
        while huff_size[p] != 0 {
            while huff_size[p] == si {
                huff_code[p] = code;
                code += 1;
                p += 1;
            }
            code <<= 1;
            si += 1;
        }
        let mut lookup = [(i32::from(HUFF_LOOKAHEAD) + 1) << HUFF_LOOKAHEAD; 1 << HUFF_LOOKAHEAD];
        let mut p = 0;
        for l in 1..=HUFF_LOOKAHEAD {
            for _ in 0..bits[usize::from(l)] {
                let mut look_bits = (huff_code[p] as usize) << (HUFF_LOOKAHEAD - l);
                for _ in 0..1 << (HUFF_LOOKAHEAD - l) {
                    lookup[look_bits] = (i32::from(l) << HUFF_LOOKAHEAD) | i32::from(values[p]);
                    look_bits += 1;
                }
                p += 1;
            }
        }
        if is_dc {
            return (lookup, None);
        }
        let mut fast = [255_i16; 1 << HUFF_LOOKAHEAD];
        for i in 0..num_symbols {
            let s = huff_size[i];
            if s <= HUFF_LOOKAHEAD {
                let c = (huff_code[i] << (HUFF_LOOKAHEAD - s)) as usize;
                for j in 0..1_usize << (HUFF_LOOKAHEAD - s) {
                    fast[c + j] = i as i16;
                }
            }
        }
        let mut fast_ac = [0_i16; 1 << HUFF_LOOKAHEAD];
        for i in 0..(1 << HUFF_LOOKAHEAD) {
            let fast_v = fast[i];
            if fast_v < 255 {
                let rs = values[fast_v as usize];
                let run = i16::from((rs >> 4) & 15);
                let mag_bits = i16::from(rs & 15);
                let len = i16::from(huff_size[fast_v as usize]);
                if mag_bits != 0 && (len + mag_bits) <= i16::from(HUFF_LOOKAHEAD) {
                    let mut k = (((i as i16) << len) & ((1 << HUFF_LOOKAHEAD) - 1))
                        >> (i16::from(HUFF_LOOKAHEAD) - mag_bits);
                    let m = 1 << (mag_bits - 1);
                    if k < m {
                        k += (!0_i16 << mag_bits) + 1;
                    }
                    if (-128..=127).contains(&k) {
                        fast_ac[i] = (k << 8) + (run << 4) + (len + mag_bits);
                    }
                }
            }
        }
        (lookup, Some(fast_ac))
    }

    /// A code-length histogram that satisfies Kraft's inequality with room to spare
    /// for the all-ones code, and as many symbols as it has codes.
    fn random_bits(state: &mut u64, max_symbols: usize) -> [u8; 17] {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        let mut bits = [0_u8; 17];
        // Codes still free at the current length, out of 2^length.
        let mut free = 1_u32;
        let mut symbols = 0;
        for l in 1..=16 {
            free *= 2;
            let room = (free - 1).min((max_symbols - symbols) as u32).min(255);
            let count = if room == 0 { 0 } else { (next() % u64::from(room + 1)) as u32 };
            // Short tables now and then fill a length completely.
            let count = if next() % 8 == 0 { room } else { count };
            bits[l] = count as u8;
            symbols += count as usize;
            free -= count;
        }
        bits
    }

    #[test]
    fn lookahead_tables_match_the_entry_by_entry_build() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut built = 0;
        // 256 codes of at most HUFF_LOOKAHEAD bits, so that the symbol at index 255
        // has entries; and every length used up to the lookahead.
        let mut edge = [[0_u8; 17]; 2];
        (edge[0][8], edge[0][9]) = (255, 1);
        (edge[1][1], edge[1][2], edge[1][3], edge[1][9]) = (1, 1, 1, 63);
        for round in 0..20_000 {
            let max_symbols = [1, 2, 12, 162, 256][round % 5];
            let bits = edge
                .get(round / 4)
                .copied()
                .unwrap_or_else(|| random_bits(&mut state, max_symbols));
            let mut values = [0_u8; 256];
            for v in &mut values {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *v = state as u8;
            }
            // Small runs and magnitudes, as real AC tables have, half of the time.
            if round % 2 == 0 {
                for v in &mut values {
                    *v &= 0x3F;
                }
            }
            for is_dc in [false, true] {
                let mut dc_values = values;
                if is_dc {
                    for v in &mut dc_values {
                        *v %= 12;
                    }
                }
                let mut slot = None;
                if HuffmanTable::build_in(&mut slot, &bits, &dc_values, is_dc, false, 15).is_err() {
                    continue;
                }
                let table = slot.unwrap();
                let (lookup, ac_lookup) = reference_lookahead(&bits, &dc_values, is_dc);
                assert_eq!(table.lookup, lookup, "lookup for {bits:?}");
                assert_eq!(table.ac_lookup, ac_lookup, "ac_lookup for {bits:?}");
                built += 1;
            }
        }
        assert!(built > 30_000, "only {built} tables were valid");
    }
}
