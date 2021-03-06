# 1. ingest readme
# 2. Iterate of sections. We will use these to create folder paths
# 3. Iterate over entries. We will create a document for each entry using the stub template. 
#    It'd be neat and fancy to create this as a jinja template, but that's like way overkill. At least for now.
#    - Create a TOC section for each top level bullet. Nest sub-docs beneath this. Some housekeeping may be necessary
#    - Create a MyST document for each reference. Let's use this pattern: YYYY_FirstAuthorFullName_arxivIDorDOIifAvailable.
# ....
# Actually, that pattern doesn't work since not everything is on arxiv or even has a link.
# Let's go with this instead: YYYY_FirstAuthorFullName_FirstKCharactersOfArticleTitle
#   

#
#    - In retrospect, if there's a way I can convert the collection I have into bibtex entries, that would be a better
#      data structure to parse for this. Oh well. Later. Bibtex + jinja. Noted.

from loguru import logger
from collections import defaultdict
from unidecode import unidecode
import string
from pathlib import Path


class Book:
    def __init__(self, 
                 readme_fpath='README.md',
                 outpath='content',
                 toc_path='_toc.md',
                 ):
        self.outpath = outpath
        self.parser = ReadmeParser(fpath=readme_fpath)
    def generate_stubs(self):
        self.stubs = []
        for topic in self.parser.entries:
            if 'subheadings' not in topic:
                continue
            for subtopic, entries in topic['subheadings'].items():
                for entry in entries:
                    stub = Stub(entry, depth=3)
                    self.stubs.append(stub)
                    # write stub to disk
                    stub.write(self.outpath)
                    
                    # map stub to TOC entry (or write TOC entry?)
                
    def add_stubs_to_toc(self):
        pass



class Stub:
    _header = """---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---"""

    _pdf_embed_template = """```{{code-cell}} ipython3
:tags: [hide-input]

import panel as pn
pn.extension()
pdf_pane = pn.pane.PDF('{pdf_url}', width=700, height=1000)
pdf_pane
```"""

    def __init__(self, item, depth=1):
        self.item = item
        self.depth=depth
    @property
    def title(self):
        h = '#'*self.depth
        return f"{h} {self.item['title_part']}"
    @property
    def pdf_embed(self):
        return self._pdf_embed_template.format(pdf_url=self.item['url'])
    def __str__(self):
        parts = [self._header]
        parts.append(self.title)
        url = self.item.get('url')
        #if self.item.get('is_pdf'):
        #    url = self.pdf_embed
        if url:
            parts.append(url)
        return '\n\n'.join(parts)
    def write(self, fpath='.', ext='.md'):
        outpath = Path(fpath) / Path(self.item['stub_name'] + ext) 
        with open(outpath, 'w') as f:
            f.write(str(self))
        
        

class ReadmeParser:
    def __init__(self, fpath='README.md'):
        self.fpath = fpath
        self.read()
        self.entries = self.parse()
        
    def read(self):
        with open(self.fpath, 'rb') as f:
            text = f.read()
        self.text = text.decode("utf-8")
        
    def parse(self):
        self.errors = []
        entries = self.parse_sections()
        entries = self.parse_entries(entries)
        entries = self.parse_subheadings(entries)
        entries = self.parse_items(entries)
        return entries
    
    def parse_sections(self):
        sections = []
        section = []
        for i, line in enumerate(self.text.split('\n')):
            if line.startswith('# '):
                if section:
                    sections.append(section)
                section = []
            section.append(line)
        return sections
    
    def parse_entries(self, sections):
        entries = []
        for i, sec in enumerate(sections):
            if i < 1:
                continue
            entry = {}
            entry['heading'], entry['rest'] = sec[0][2:].strip(), sec[1:]
            entries.append(entry)
        return entries
    
    def parse_subheadings(self, entries):
        subheading = ""
        for entry in entries:
            subheadings = defaultdict(list)
            for line in entry['rest']:
                if line.startswith('* '):
                    subheading = line.strip()[2:]
                    continue
                if subheading:
                    line = line.strip()
                    if line:
                        subheadings[subheading].append(line)
            entry.pop('rest')
            if subheadings:
                entry['subheadings'] = subheadings
        return entries
                
    def parse_year(self, item):
        rec = {}
        parts = item.split('-')
        rec['year_part'], rec['rest'] = parts[0], parts[1:]
        rec['year'] = int(rec['year_part'][1:].strip())
        rec.pop('year_part')
        return rec
    def parse_title(self, rec):
        parts = '-'.join(rec['rest']).split('](')
        if len(parts) <2:
            # continue # hmmm......
            raise ValueError # this part wasn't writing into an error before. should it be now?
        if len(parts) > 2:
            #errors.append(item)
            #continue
            raise ValueError # I guess?
        rec['title_part'], rec['rest'] = parts
        rec['title_part'] = rec['title_part'].strip()[1:] # remove starting bracket. need to standardize/remove quotes
        return rec
    
    def parse_url_and_authors(self, rec):
        parts = rec['rest'].split(') -') 

        if len(parts) <2:
            #continue
            raise ValueError # I guess?
        if len(parts) > 2:
            #errors.append(item)
            #continue
            raise ValueError # I guess?
        rec['url'], rec['authors_part'] = parts
        rec.pop('rest')

        if rec['url'].endswith('.pdf'):
            rec['is_pdf'] = 'True'

        rec['authors'] = [a.strip() for a in rec['authors_part'].split(',')]
        rec.pop('authors_part')
        return rec
        
    def parse_items(self, entries):
        errors = self.errors
        
        for entry in entries:
            if 'subheadings' not in entry:
                continue # actually I want to remove these, but deal with it later
            for topic, items in entry['subheadings'].items():
                recs = []
                for item in items:
                    
                    try:
                        rec = self.parse_year(item)
                        rec = self.parse_title(rec)
                        rec = self.parse_url_and_authors(rec)
                        rec['stub_name'] = self.generate_fname(rec)
                    except ValueError:
                        logger.warning(item)
                        errors.append(item)
                    recs.append(rec)

                if recs:
                    entry['subheadings'][topic] = recs
        return entries
    def generate_fname(self, entry, k_title_chars=30):
        """YYYY_FirstAuthorFullName_FirstKCharactersOfArticleTitle"""
        def clean(text):
            s = text.title().replace(' ', '')
            # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
            s = s.translate(str.maketrans('', '', string.punctuation))
            return unidecode(s)
        author = entry['authors'][0]
        author_clean = clean(author)
        title_clean = clean(entry['title_part'])
        return f"{entry['year']}_{author_clean}_{title_clean[:k_title_chars]}"
        
# to do:
# - generate stubs from items
# - generate TOC entries
        
if __name__ == '__main__':
    parser = ReadmeParser()
    stub_path='anthology_of_modern_ml/stubs'
    for entry in parser.entries:
        if 'subheadings' not in entry:
            continue
        for subheading, items in entry['subheadings'].items():
            logger.debug(f'subheading: {subheading}')
            for item in items:
                logger.debug(f'item:\n {item}')
                stub = Stub(item)
                try:
                    stub.write(stub_path)
                except Exception as e:
                    logger.warning(e)