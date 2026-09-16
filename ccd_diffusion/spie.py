"""
The front and back matter of an SPIE journal article, written against the
``spieman`` class, for use with the journal-independent parts of :mod:`aastex`.

The class expects the ``authblk`` conventions, ``\\author[a,*]{}`` and
``\\affil[a]{}``, an ``abstract`` environment, a ``\\keywords{}`` command
after the abstract, and unnumbered sections for the disclosures, the data
availability statement, and the acknowledgments before the references.
"""

import dataclasses
import pathlib
import pylatex

__all__ = [
    "path_class",
    "path_bibliography_style",
    "Title",
    "Affiliation",
    "Author",
    "Abstract",
    "Keywords",
    "Corresponding",
    "Disclosures",
    "Availability",
    "Acknowledgments",
    "Biography",
]

path_class = pathlib.Path(__file__).parent / "spieman.cls"
"""The SPIE journal class file distributed with this article."""

path_bibliography_style = pathlib.Path(__file__).parent / "spiejour.bst"
"""The SPIE journal bibliography style distributed with this article."""


@dataclasses.dataclass
class Title(pylatex.base_classes.LatexObject):
    """The title of the article."""

    name: str
    """The title."""

    def dumps(self) -> str:
        return pylatex.Command("title", pylatex.NoEscape(self.name)).dumps()


@dataclasses.dataclass
class Affiliation(pylatex.base_classes.LatexObject):
    """An organization that an author is associated with."""

    key: str
    """The letter used to mark the authors with this affiliation, ``"a"``, ``"b"``, ..."""

    name: str
    """The name and address of the organization."""

    def dumps(self) -> str:
        return pylatex.Command(
            command="affil",
            arguments=pylatex.NoEscape(self.name),
            options=self.key,
        ).dumps()


@dataclasses.dataclass
class Author(pylatex.base_classes.LatexObject):
    """One of the authors of the article."""

    name: str
    """The name of the author."""

    affiliation: Affiliation | list[Affiliation]
    """The organizations affiliated with the author."""

    email: None | str = None
    """The email address of the author, shown only for the corresponding author."""

    orcid: None | str = None
    """The ORCID of the author, which SPIE collects at submission rather than in the manuscript."""

    corresponding: bool = False
    """Whether this author is the corresponding author, marked with an asterisk."""

    @property
    def affiliations(self) -> list[Affiliation]:
        """The affiliations as a list, whether one or several were given."""
        if isinstance(self.affiliation, Affiliation):
            return [self.affiliation]
        return list(self.affiliation)

    def dumps(self) -> str:
        keys = [a.key for a in self.affiliations]
        if self.corresponding:
            keys.append("*")
        return pylatex.Command(
            command="author",
            arguments=pylatex.NoEscape(self.name),
            options=",".join(keys),
        ).dumps()


class Abstract(pylatex.base_classes.Environment):
    """
    The abstract of the article, a single self-contained paragraph with no
    citations.
    """

    _latex_name = "abstract"
    escape = False


@dataclasses.dataclass
class Keywords(pylatex.base_classes.LatexObject):
    """The three to six keywords which follow the abstract."""

    keywords: list[str]
    """The keywords."""

    def __post_init__(self):
        if not 3 <= len(self.keywords) <= 6:
            raise ValueError("SPIE journals ask for 3 to 6 keywords")

    def dumps(self) -> str:
        return pylatex.Command(
            "keywords", pylatex.NoEscape(", ".join(self.keywords))
        ).dumps()


@dataclasses.dataclass
class Corresponding(pylatex.base_classes.LatexObject):
    """
    The line naming the corresponding author and their email address, which
    the class does not generate itself and which the template places just
    after the keywords.
    """

    author: Author
    """The corresponding author."""

    def dumps(self) -> str:
        email = self.author.email
        if email is None:
            raise ValueError("the corresponding author needs an email address")
        return (
            r"\noindent{\footnotesize *"
            + self.author.name
            + ", "
            + r"\href{mailto:"
            + email
            + "}{"
            + email
            + "}}"
        )


def _unnumbered(title: str, text: str) -> pylatex.Subsection:
    result = pylatex.Subsection(title, numbering=False)
    result.escape = False
    result.append(pylatex.NoEscape(text))
    return result


def Disclosures(text: str) -> pylatex.Subsection:
    """
    The unnumbered section declaring conflicts of interest and the use of
    any AI tools.

    Parameters
    ----------
    text
        The statement.
    """
    return _unnumbered("Disclosures", text)


def Availability(text: str) -> pylatex.Subsection:
    """
    The unnumbered section describing how to obtain the code, data, and
    materials behind the article.

    Parameters
    ----------
    text
        The statement.
    """
    return _unnumbered("Code, Data, and Materials Availability", text)


class Acknowledgments(pylatex.base_classes.Environment):
    """The acknowledgments, which the class typesets as an unnumbered section."""

    _latex_name = "acknowledgments"
    escape = False


@dataclasses.dataclass
class Biography(pylatex.base_classes.LatexObject):
    """
    A professional biography of about 75 words, placed after the references.
    """

    author: Author
    """The author being described."""

    text: str
    """The biography."""

    def dumps(self) -> str:
        return (
            r"\vspace{2ex}\noindent\textbf{"
            + self.author.name
            + "} "
            + self.text.strip()
        )
