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
        
if __name__ == '__main__':
    parser = ReadmeParser()