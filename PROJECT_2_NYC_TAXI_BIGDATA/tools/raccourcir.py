"""Remplace le texte des cellules Markdown d'une section, sans toucher au code."""
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent


def blocs(source):
    """Decoupe le fichier en blocs md(...) et code(...) dans l'ordre."""
    motif = re.compile(r'(md|code)\((""".*?"""|\'\'\'.*?\'\'\')\),', re.S)
    return [(m.start(), m.end(), m.group(1), m.group(2)) for m in motif.finditer(source)]


def lister(fichier):
    s = Path(fichier).read_text()
    for i, (_, _, genre, contenu) in enumerate(blocs(s)):
        if genre == "md":
            titre = next((l for l in contenu.split("\n") if l.strip().startswith("#")),
                         contenu.strip('"\n').split("\n")[0])
            print(f"[{i:2d}] md   {titre.strip()[:70]}")
        else:
            premiere = next((l for l in contenu.split("\n")[1:] if l.strip()), "")
            print(f"[{i:2d}] code {premiere.strip()[:70]}")


def remplacer(fichier, remplacements):
    """remplacements : {index_du_bloc: nouveau_texte_markdown}"""
    p = Path(fichier)
    s = p.read_text()
    for i in sorted(remplacements, reverse=True):
        debut, fin, genre, _ = blocs(s)[i]
        assert genre == "md", f"Le bloc {i} est du code, pas du markdown"
        nouveau = remplacements[i]
        s = s[:debut] + 'md("""\n' + nouveau.strip("\n") + '\n"""),' + s[fin:]
    p.write_text(s)


def supprimer(fichier, indices):
    p = Path(fichier)
    s = p.read_text()
    for i in sorted(indices, reverse=True):
        debut, fin, _, _ = blocs(s)[i]
        reste = s[fin:]
        saut = len(reste) - len(reste.lstrip("\n"))
        s = s[:debut] + reste[saut:]
    p.write_text(s)


if __name__ == "__main__":
    lister(sys.argv[1])
