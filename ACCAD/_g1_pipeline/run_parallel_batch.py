#!/usr/bin/env python3
"""Process-sharded entry point for the canonical ACCAD-to-G1 batch."""

from accad_g1.parallel_batch import main


if __name__ == "__main__":
    raise SystemExit(main())
