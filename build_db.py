from collections import defaultdict
import json
from readme2book import ReadmeParser


def collect_docs(parser):
    items = []
    for entry in parser.entries:
        heading = entry['heading']
        if 'subheadings' not in entry: # yeesh...
            continue
        for subheading, item in entry['subheadings'].items():
            #stub = Stub(item)
            tags=[heading, subheading]
            for rec in item:
                if not rec.get('tags'):
                    rec['tags'] = tags
            items.extend(item)
    return {rec['stub_name']:rec for rec in items if rec.get('stub_name')}


def aggregate(docs, key):
    d_ = defaultdict(list)
    for doc_id, doc in docs.items():
        record = doc[key]
        if isinstance(record, list):
            for item in record:
                d_[item].append(doc_id)
        else:
            d_[record].append(doc_id)
    return dict(sorted((d_.items())))


def collect_partial_parses(parser):
    partial_parses=[]
    for entry in parser.entries:
        if 'subheadings' not in entry:
            continue
        for subheading, item in entry['subheadings'].items():
            for rec in item:
                if not rec.get('stub_name'):
                    partial_parses.append(rec)
    return partial_parses


if __name__ == '__main__':
    
    parser = ReadmeParser()
    partial_parses = collect_partial_parses(parser)
    docs = collect_docs(parser)
    
    authors = aggregate(docs, 'authors')
    tags = aggregate(docs, 'tags')
    years = aggregate(docs, 'year')
    
    db=dict(
        docs=docs,
        authors=authors,
        tags=tags,
        years=years,
        errors=dict(
            parser_errors=parser.errors,
            partial_parses=partial_parses,
        ),
    )
    
    with open("db.json", 'w') as f:
        json.dump(db, f)