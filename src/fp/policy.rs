//! The parameter file (`frame/parameters.h`, `config/policy.txt`):
//! one order, one spelling. Line-based: a keyword, an optional name for the
//! `group`/`stages`/`work` families, then integers. First line `version H821`.
//! Parser-enforced rules from FORMAT.md: `listed` is taken at 0 alone,
//! `answers` at 1 alone. Unknown keywords are stored, not rejected — planning
//! consumes them in later increments, and every byte of this file enters the
//! configuration digest anyway. Malformed lines fail closed.

use alloc::{string::String, vec::Vec};

use crate::fp::{
    error::FpError,
    version::{Version, H821},
};

/// Keywords whose lines carry a name before the integers
/// (`group link …`, `stages link …`, `work positions …`).
const NAMED_KEYWORDS: [&str; 3] = ["group", "stages", "work"];

/// Layout call lines (`call R16a R16 G U`): keyword, call name, target,
/// then symbolic argument names — no integers.
const CALL_KEYWORDS: [&str; 3] = ["call", "callA", "callH"];

/// One layout call line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallLine {
    pub keyword: String,
    pub name: String,
    pub target: String,
    pub symbols: Vec<String>,
}

/// One parameter line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyLine {
    pub keyword: String,
    pub name: Option<String>,
    pub args: Vec<u64>,
}

/// A parsed parameter file.
#[derive(Debug, Clone)]
pub struct Policy {
    pub version: Version,
    pub lines: Vec<PolicyLine>,
    pub calls: Vec<CallLine>,
}

impl Policy {
    pub fn parse(bytes: &[u8]) -> Result<Self, FpError> {
        let text = core::str::from_utf8(bytes).map_err(|_| FpError::NotUtf8)?;
        let mut lines: Vec<PolicyLine> = Vec::new();
        let mut calls: Vec<CallLine> = Vec::new();
        let mut version = None;
        for (index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            let mut words = line.split_whitespace();
            let keyword = words.next().ok_or(FpError::Policy {
                context: "empty-line",
            })?;
            if index == 0 {
                if keyword != "version" {
                    return Err(FpError::Policy {
                        context: "version-first-line",
                    });
                }
                let word = words.next().ok_or(FpError::Policy {
                    context: "version-missing",
                })?;
                version = Some(Version::parse(word.as_bytes())?);
                if words.next().is_some() {
                    return Err(FpError::Policy {
                        context: "version-extra-words",
                    });
                }
                continue;
            }
            if keyword == "version" {
                return Err(FpError::Policy {
                    context: "version-repeated",
                });
            }
            if CALL_KEYWORDS.contains(&keyword) {
                let name = words.next().ok_or(FpError::Policy {
                    context: "call-name",
                })?;
                let target = words.next().ok_or(FpError::Policy {
                    context: "call-target",
                })?;
                let call = CallLine {
                    keyword: String::from(keyword),
                    name: String::from(name),
                    target: String::from(target),
                    symbols: words.map(String::from).collect(),
                };
                if calls
                    .iter()
                    .any(|c| c.keyword == call.keyword && c.name == call.name)
                {
                    return Err(FpError::Policy {
                        context: "duplicate-call",
                    });
                }
                calls.push(call);
                continue;
            }
            let mut args = Vec::new();
            let mut name = None;
            let mut rest = words.peekable();
            if NAMED_KEYWORDS.contains(&keyword) {
                let named = rest.next().ok_or(FpError::Policy {
                    context: "name-missing",
                })?;
                name = Some(String::from(named));
            }
            for word in rest {
                let value = word.parse::<u64>().map_err(|_| FpError::Policy {
                    context: "integer-argument",
                })?;
                args.push(value);
            }
            let new_line = PolicyLine {
                keyword: String::from(keyword),
                name,
                args,
            };
            if lines
                .iter()
                .any(|l| l.keyword == new_line.keyword && l.name == new_line.name)
            {
                return Err(FpError::Policy {
                    context: "duplicate-keyword",
                });
            }
            lines.push(new_line);
        }
        let version = version.ok_or(FpError::Policy {
            context: "version-missing",
        })?;
        let policy = Self {
            version,
            lines,
            calls,
        };
        policy.require_singleton("listed", 0)?;
        policy.require_singleton("answers", 1)?;
        policy.require_present("sec")?;
        policy.require_present("salt")?;
        if let Some(record) = policy.get("record") {
            if record.len() != 1 || record[0] > 1 {
                return Err(FpError::Policy {
                    context: "record-bool",
                });
            }
        }
        Ok(policy)
    }

    /// Integer arguments of the first line with this keyword.
    pub fn get(&self, keyword: &str) -> Option<&[u64]> {
        self.lines
            .iter()
            .find(|l| l.keyword == keyword)
            .map(|l| l.args.as_slice())
    }

    /// Integer arguments of a named line (`group link`, `work positions`, …).
    pub fn get_named(&self, keyword: &str, name: &str) -> Option<&[u64]> {
        self.lines
            .iter()
            .find(|l| l.keyword == keyword && l.name.as_deref() == Some(name))
            .map(|l| l.args.as_slice())
    }

    fn require_present(&self, keyword: &'static str) -> Result<(), FpError> {
        if self.get(keyword).is_none() {
            return Err(FpError::Policy { context: keyword });
        }
        Ok(())
    }

    fn require_singleton(&self, keyword: &'static str, expected: u64) -> Result<(), FpError> {
        match self.get(keyword) {
            Some([value]) if *value == expected => Ok(()),
            _ => Err(FpError::Policy { context: keyword }),
        }
    }

    pub fn salt_len(&self) -> Option<u64> {
        self.get("salt").and_then(|args| args.first().copied())
    }

    pub fn sec(&self) -> Option<u64> {
        self.get("sec").and_then(|args| args.first().copied())
    }

    pub fn record(&self) -> bool {
        matches!(self.get("record"), Some([1]))
    }

    pub fn h821(&self) -> bool {
        self.version == H821
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = "version H821\nsec 128\nrecord 1\nwork positions 20 28 32\nsalt 40\nlisted 0\ngroup link 1 18 17 17 13\ngroup folded 1 17 15 13 6\nstages link 4 3 17 17 3 3 16 16\nanswers 1\n";

    #[test]
    fn parses_real_shaped_policy() {
        let policy = Policy::parse(GOOD.as_bytes()).unwrap();
        assert!(policy.h821());
        assert_eq!(policy.sec(), Some(128));
        assert_eq!(policy.salt_len(), Some(40));
        assert!(policy.record());
        assert_eq!(
            policy.get_named("work", "positions"),
            Some(&[20, 28, 32][..])
        );
        assert_eq!(
            policy.get_named("group", "link"),
            Some(&[1, 18, 17, 17, 13][..])
        );
        assert_eq!(
            policy.get_named("group", "folded"),
            Some(&[1, 17, 15, 13, 6][..])
        );
        assert_eq!(
            policy.get_named("stages", "link"),
            Some(&[4, 3, 17, 17, 3, 3, 16, 16][..])
        );
        assert_eq!(policy.get_named("group", "own"), None);
    }

    #[test]
    fn rejects_missing_or_late_version() {
        assert!(Policy::parse(b"sec 128\n").is_err());
        assert!(Policy::parse(b"sec 128\nversion H821\n").is_err());
        assert!(Policy::parse(b"version H820\nsec 128\nsalt 40\nlisted 0\nanswers 1\n").is_err());
    }

    #[test]
    fn rejects_listed_nonzero_and_answers_not_one() {
        assert!(Policy::parse(b"version H821\nsec 128\nsalt 40\nlisted 4\nanswers 1\n").is_err());
        assert!(Policy::parse(b"version H821\nsec 128\nsalt 40\nlisted 0\nanswers 2\n").is_err());
    }

    #[test]
    fn rejects_bad_record() {
        assert!(
            Policy::parse(b"version H821\nsec 128\nsalt 40\nlisted 0\nanswers 1\nrecord 3\n")
                .is_err()
        );
    }

    #[test]
    fn rejects_duplicates_and_bad_integers() {
        assert!(
            Policy::parse(b"version H821\nsec 128\nsec 128\nsalt 40\nlisted 0\nanswers 1\n")
                .is_err()
        );
        assert!(Policy::parse(b"version H821\nsec abc\nsalt 40\nlisted 0\nanswers 1\n").is_err());
        assert!(Policy::parse(b"version H821\nsec -1\nsalt 40\nlisted 0\nanswers 1\n").is_err());
        assert!(Policy::parse(b"version H821\nsec 128\nsalt 40\nlisted 0\nanswers 1\ngroup link 1 2\ngroup link 3 4\n").is_err(), "same (keyword, name) twice must fail");
    }

    #[test]
    fn distinct_named_groups_are_not_duplicates() {
        assert!(Policy::parse(GOOD.as_bytes()).is_ok());
    }

    #[test]
    fn rejects_non_utf8() {
        assert!(matches!(
            Policy::parse(&[0xff, 0xfe]),
            Err(FpError::NotUtf8)
        ));
    }

    #[test]
    fn parses_call_lines() {
        let policy = Policy::parse(b"version H821\nsec 128\nsalt 40\nlisted 0\nanswers 1\ncall R16a R16 G U\ncallA WH WH WH\n").unwrap();
        assert_eq!(policy.calls.len(), 2);
        assert_eq!(policy.calls[0].name, "R16a");
        assert_eq!(policy.calls[0].target, "R16");
        assert_eq!(policy.calls[0].symbols, ["G", "U"]);
        assert_eq!(policy.calls[1].symbols, ["WH"]);
    }

    #[test]
    fn rejects_duplicate_and_incomplete_calls() {
        assert!(Policy::parse(b"version H821\nsec 128\nsalt 40\nlisted 0\nanswers 1\ncall R16a R16 G\ncall R16a R16 U\n").is_err(), "duplicate (keyword, name) must fail");
        assert!(
            Policy::parse(b"version H821\nsec 128\nsalt 40\nlisted 0\nanswers 1\ncall R16a\n")
                .is_err(),
            "missing target must fail"
        );
    }
}
