"""Typed exception hierarchy — precise, actionable failures."""

from __future__ import annotations


class GenomeFirewallError(Exception):
    """Base for every error this package raises."""


class BVBRCError(GenomeFirewallError):
    """A BV-BRC API request failed after retries."""


class AMRFinderError(GenomeFirewallError):
    """AMRFinderPlus failed to run or produced no parseable output.

    Most common cause: the `amr` micromamba env is missing. Install hint:
        ~/bin/micromamba create -n amr -c bioconda -c conda-forge ncbi-amrfinderplus mash
        ~/bin/micromamba run -n amr amrfinder -u   # download the database
    """


class EmptyFastaError(GenomeFirewallError):
    """A genome had no sequence returned from the API (bad/withdrawn genome_id)."""


class UnknownDrugError(GenomeFirewallError):
    """A requested drug is not on the panel (typo/miscasing/unsupported antibiotic)."""


class InsufficientDataError(GenomeFirewallError):
    """Too few labels/positives to train or calibrate a drug honestly → drug-level no-call."""
