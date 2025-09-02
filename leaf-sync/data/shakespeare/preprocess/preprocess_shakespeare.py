#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesses the Shakespeare dataset for federated training.

Uso:
  python preprocess_shakespeare.py path/to/raw_data.txt output_directory/

Produce:
  - output_directory/users_and_plays.json
  - output_directory/by_play_and_character/<PLAY>_<CHAR>.txt
"""

import os
import re
import sys
import json
import unicodedata
import collections

# Marcadores robustos para el texto de Gutenberg
AUTHOR_RE = re.compile(r'by william shakespeare', re.IGNORECASE)
END_RE    = re.compile(r'end of the project gutenberg ebook', re.IGNORECASE)

# -----------------------------------------------------------
# Normalización de texto para evitar caracteres fuera del vocabulario (índices -1)
# -----------------------------------------------------------
def _normalize_snippet(s: str) -> str:
    # Comillas curvas → rectas
    s = s.replace('’', "'").replace('‘', "'")
    s = s.replace('“', '"').replace('”', '"')
    # Guiones largos → guiones simples
    s = s.replace('—', '-').replace('–', '-').replace('•', '-')
    # NBSP → espacio normal
    s = s.replace('\u00A0', ' ')
    # Normalización Unicode
    s = unicodedata.normalize('NFKC', s)
    # Mantener ASCII imprimible
    s = ''.join(ch if 32 <= ord(ch) < 127 else ' ' for ch in s)
    # Colapsar espacios múltiples
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# -----------------------------------------------------------
# Parseo del texto completo → lista de obras con parlamentos por personaje
# -----------------------------------------------------------
def _split_into_plays(shakespeare_full):
    """
    Devuelve:
      plays: [(title, {CHAR: [linea, ...], ...}), ...]
      discarded_lines: [str, ...] (solo informativo)
    """
    plays = []
    discarded_lines = []

    slines = shakespeare_full.splitlines(True)  # conservar saltos

    # Cortar encabezado hasta la 1ª aparición de "by William Shakespeare"
    author_hits = [i for i, l in enumerate(slines) if AUTHOR_RE.search(l)]
    if author_hits:
        start_i = max(author_hits[0] - 7, 0)
        slines = slines[start_i:]

    current_character = None
    comedy_of_errors = False
    characters = None
    title = None

    for i, line in enumerate(slines):
        # Fin del libro (marcador estándar de Gutenberg)
        if END_RE.search(line):
            break

        # Heurística de inicio de nueva obra
        if AUTHOR_RE.search(line):
            # cerrar obra anterior (si existía y tenía parlamentos)
            if characters:
                # filtrar obras sin parlamentos
                if any(len(v) > 0 for v in characters.values()):
                    plays.append((title, characters))

            # nueva obra
            current_character = None
            characters = collections.defaultdict(list)

            # el título suele estar unas líneas arriba del "by William Shakespeare"
            title_candidate = ""
            for back in (2, 3, 4, 5, 6, 7):
                j = i - back
                if j >= 0:
                    t = slines[j].strip()
                    if t:
                        title_candidate = t
                        break
            title = title_candidate if title_candidate else f"UNKNOWN_PLAY_{len(plays)}"
            comedy_of_errors = (title.strip().upper() == 'THE COMEDY OF ERRORS')
            continue

        # Línea de personaje (permitir 0–4 espacios antes de "NAME. texto")
        m = re.match(r'^\s{0,4}([A-Za-z][A-Za-z ]*)\. (.*)', line)
        if m and characters is not None:
            character, snippet = m.group(1), m.group(2)
            character = character.upper().strip()
            # evitar “ACT I/II...” como personaje en Comedy of Errors (compatibilidad)
            if not (comedy_of_errors and character.startswith('ACT ')):
                characters[character].append(_normalize_snippet(snippet))
                current_character = character
            else:
                current_character = None
            continue

        # Continuación de parlamento (≥4 espacios)
        if current_character and characters is not None:
            m2 = re.match(r'^\s{4,}(.*)', line)
            if m2:
                snip = m2.group(1)
                # caso especial del original
                if comedy_of_errors and snip.startswith('<'):
                    current_character = None
                    continue
                characters[current_character].append(_normalize_snippet(snip))
                continue

        # Líneas que no clasifican (solo para log opcional)
        ls = line.strip()
        if ls:
            discarded_lines.append(f'{i}:{ls}')

    # cerrar última obra
    if characters:
        if any(len(v) > 0 for v in characters.values()):
            plays.append((title, characters))

    # filtrar obras sin parlamentos útiles
    plays = [(t, ch) for (t, ch) in plays if len(ch) > 1]
    return plays, discarded_lines

# -----------------------------------------------------------
# Convertir plays → ejemplos por personaje y diccionario users_and_plays
# -----------------------------------------------------------
def _sanitize_name(s: str) -> str:
    s = _normalize_snippet(s)
    s = re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')
    return s

def _get_train_test_by_character(plays, test_fraction=-1.0):
    """
    Crea:
      users_and_plays: { "<PLAY>_<CHAR>": "<PLAY>", ... }
      all_examples:    { "<PLAY>_<CHAR>": [ "texto...", ... ] }
      test_examples:   {} (no se usa aquí; test_fraction=-1.0)
    """
    users_and_plays = {}
    all_examples = {}
    test_examples = {}

    for (title, characters) in plays:
        play_name = _sanitize_name(title)
        for char, bites in characters.items():
            if not bites:
                continue
            user = f"{play_name}_{_sanitize_name(char)}"
            # juntar parlamentos en “pasajes” (1 línea por parlamento normalizado)
            examples = [ln for ln in (bites or []) if ln]
            if not examples:
                continue
            users_and_plays[user] = play_name
            all_examples[user] = examples

    return users_and_plays, all_examples, test_examples

# -----------------------------------------------------------
# Escritura de archivos por personaje
# -----------------------------------------------------------
def _write_data_by_character(examples, output_directory):
    """Escribe un .txt por usuario (<PLAY>_<CHAR>.txt) con sus parlamentos."""
    os.makedirs(output_directory, exist_ok=True)
    for character_name, sound_bites in examples.items():
        filename = os.path.join(output_directory, character_name + '.txt')
        with open(filename, 'w', encoding='utf-8') as output:
            for sound_bite in sound_bites:
                output.write(sound_bite + '\n')

# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
def main(argv):
    if len(argv) < 2:
        print("Uso: python preprocess_shakespeare.py <raw_data.txt> <output_dir>", file=sys.stderr)
        sys.exit(1)

    print('Splitting .txt data between users')
    input_filename = argv[0]
    output_directory = argv[1]

    with open(input_filename, 'r', encoding='utf-8', errors='ignore') as input_file:
        shakespeare_full = input_file.read()

    plays, discarded_lines = _split_into_plays(shakespeare_full)
    print('Discarded %d lines' % len(discarded_lines))

    users_and_plays, all_examples, _ = _get_train_test_by_character(
        plays, test_fraction=-1.0
    )

    with open(os.path.join(output_directory, 'users_and_plays.json'), 'w', encoding='utf-8') as ouf:
        json.dump(users_and_plays, ouf)

    _write_data_by_character(
        all_examples,
        os.path.join(output_directory, 'by_play_and_character/')
    )

if __name__ == '__main__':
    main(sys.argv[1:])
