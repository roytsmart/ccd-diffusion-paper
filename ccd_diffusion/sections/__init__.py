"""
The prose of the article, one module per section.
"""

from ._s0_abstract import abstract
from ._s1_introduction import introduction
from ._s2_method import method
from ._s3_results import results
from ._s4_discussion import discussion
from ._s5_conclusion import conclusion
from ._a1_gallery import gallery
from ._s6_back_matter import disclosures, availability, acknowledgments

__all__ = [
    "abstract",
    "introduction",
    "method",
    "results",
    "discussion",
    "conclusion",
    "gallery",
    "disclosures",
    "availability",
    "acknowledgments",
]
