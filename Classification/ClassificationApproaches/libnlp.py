import spacy
from pathlib import Path
from spacy import displacy
import textacy as tx
from spacy.matcher import Matcher
import re
from svgwrite import Drawing, rgb

verb_patterns_for_verb_phrases = [
    [{"POS": "AUX"}, {"POS": "VERB"}, {"POS": "ADP"}],
    [{"POS": "AUX"}, {"POS": "VERB"}],
    [{"POS": "VERB"}]
]


def contains_root(verb_phrase, root):
    vp_start = verb_phrase.start  # get the start
    vp_end = verb_phrase.end  # and end of the phrase
    # if the root is with the start and end of the phrase then it contains the root.
    if (root.i >= vp_start and root.i <= vp_end):
        return True
    else:
        return False


def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token = token
    return root_token


def get_verb_phrases_textacy(doc):
    root = find_root_of_sentence(doc)
    verb_phrases = tx.extract.matches.token_matches(
        doc, verb_patterns_for_verb_phrases)  # returns a list of spans
    new_vps = []
    for verb_phrase in verb_phrases:
        print(type(verb_phrase))  # Output: <class 'str'>

        if (contains_root(verb_phrase, root)):
            new_vps.append(verb_phrase)
        else:
            print(
                f"We do not consider the clause: {verb_phrase} because it does not reference the root")
    return new_vps


def get_verb_phrases(nlp, doc):
    verb_phrase_pattern = [
        {"POS": {"IN": ["VERB", "AUX"]}},
        # ?	Make the pattern optional, by allowing it to match 0 or 1 times.
        {"POS": {"IN": ["VERB", "AUX"]}, "OP": "?"},
        # *	Allow the pattern to match 0 or more times.
        {"POS": {"IN": ["ADV", "PART", "ADP"]}, "OP": "*"}
    ]
    matcher = Matcher(nlp.vocab)
    matcher.add("VERB_PHRASES", [
                verb_phrase_pattern], greedy="LONGEST")  # By setting greedy="LONGEST", the matcher will prefer longer matches over shorter ones. It means that if there are multiple patterns that could potentially match the same tokens, the matcher will select the longest matching pattern. This ensures that the matcher tries to find the most specific and comprehensive matches.
    matches = matcher(doc)
    matches.sort(key=lambda x: x[1])
    root = find_root_of_sentence(doc)
    # print(len(matches))
    new_vps = []
    for match in matches[:10]:
        # print(type(match))
        #print(match, doc[match[1]:match[2]])
        verb_phrase = doc[match[1]:match[2]]  # create a span
        new_vps.append(verb_phrase)
        # if (contains_root(verb_phrase, root)):
        #     new_vps.append(verb_phrase)
        # else:
        #     print(
        #         f"We do not consider the verb in : {verb_phrase} because it does not reference the root")
    return new_vps

# The longer_verb_phrase function finds the longest verb phrase


def longer_verb_phrase(verb_phrases):
    longest_length = 0
    longest_verb_phrase = None
    for verb_phrase in verb_phrases:
        if len(verb_phrase) > longest_length:
            longest_verb_phrase = verb_phrase
    return longest_verb_phrase


def find_noun_phrase(verb_phrase, noun_phrases, side):
    for noun_phrase in noun_phrases:
        print(f"Noun Phrase: {noun_phrase} start {noun_phrase.start}")
        if (side == "left" and noun_phrase.start < verb_phrase.start):
            return noun_phrase
        elif (side == "right" and noun_phrase.start > verb_phrase.start):
            return noun_phrase

# Returns the left noun phrase, verb phrase and the right noun phrase


def find_triplet(doc, nlp=None):
    verb_phrases = list(get_verb_phrases(nlp, doc))
    #verb_phrases = list(get_verb_phrases(doc))
    phrases = []

    verb_phrase = None
    # if (len(verb_phrases) > 1):
    #     verb_phrase = longer_verb_phrase(list(verb_phrases))
    # else:
    #     verb_phrase = verb_phrases[0]
    for verb_phrase in verb_phrases:
        noun_phrases = doc.noun_chunks
        left_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "left")
        right_noun_phrase = find_noun_phrase(
            verb_phrase, noun_phrases, "right")
        phrases.append((left_noun_phrase, verb_phrase, right_noun_phrase))
    # return (left_noun_phrase, verb_phrase, right_noun_phrase)
    return phrases


def show_sentence_parts(doc):
    print(doc)
    print("{:<12} | {:<6} | {:<8} | {:<8} | {:<24} | {:<20} | {:<10} ".format(
        'Text', 'Index', 'POS', 'Dep', 'Dep Detail', 'Ancestors', 'Children'))
    print("----------------------------------------------------------------------------------------------------------------------")
    for token in doc:
        # the term "ancestors" refers to the set of nodes that are higher in the parse tree hierarchy and lead to the current token or span of tokens.
        ancestors = ' '.join([t.text for t in token.ancestors])
        # "children" refer to the nodes that are directly dependent on the current token or span of tokens in the parse tree. Children can be thought of as the "child" nodes that are connected to the current node.
        children = ' '.join([t.text for t in token.children])

        print("{:<12} | {:<6} | {:<8} | {:<8} | {:<24} | {:<20} | {:<10} ".format(
            token.text, token.i, token.pos_, token.dep_, spacy.explain(token.dep_), ancestors, children))
        print("----------------------------------------------------------------------------------------------------------------------")


def show_noun_chunks(doc):
    print("Noun Chunks\n")
    for noun_chunk in doc.noun_chunks:
        print(f"{noun_chunk.text}  Start: {noun_chunk.start} End: {noun_chunk.end} Root: {noun_chunk.root}")


def show_sentence_structure(doc):

    print("-------------------------")
    for token in doc:
        print(
            f"Token: {token.text} POS: {token.pos_} - [{spacy.explain(token.pos_)}]  Dependencies: {token.dep_} - [{spacy.explain(token.dep_)}] Fine Grained Tag: {token.tag_} - [{spacy.explain(token.tag_)}]")
    for chunk in doc.noun_chunks:
        print(f" Chunk: {chunk.text.lower()}")
    print("-------------------------")


def filter_verb_children(children):
    filtered_children = []
    for child in children:
        if child.dep_ == "advcl":
            continue
        else:
            filtered_children.append(child)
    return filtered_children


def get_clause_token_span_for_verb(verb, doc, all_verbs):
    first_token_index = len(doc)
    last_token_index = 0
    this_verb_children = filter_verb_children(list(verb.children))
    for child in this_verb_children:
        if (child not in all_verbs):
            if (child.i < first_token_index):
                first_token_index = child.i
            if (child.i > last_token_index):
                last_token_index = child.i
    return(first_token_index, last_token_index)

# The verb with a dependency of ROOT. The top of the syntactic tree


def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token = token
    return root_token


def find_other_verbs(doc, root_token):
    other_verbs = []
    for token in doc:
        ancestors = list(token.ancestors)
        if (token.pos_ == "VERB" and len(ancestors) == 1  # if it is a verb and has one ancestor which is the root token
                and ancestors[0] == root_token):
            other_verbs.append(token)
    return other_verbs

# you may want to add:
# <rect fill="rgb(255,255,255)" height="100%" width="100%" x="0px" y="0px" />
# for a white background


def render_ent(doc):
    # Create the HTML markup with colors for different word types
    html_markup = displacy.render(doc, style="ent", options={"colors": {
                                  "NOUN": "#ff0000", "VERB": "#00ff00"}})
    # Create a simple HTML file to display the text with colors
    file_name = 'ent.html'
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html_markup)


def render(doc):
    svg = displacy.render(doc, style="dep", jupyter=False, options={
        "bg": "#00ff00", "add_lemma": True, "fine_grained": True, "add_dep": True, "collapse_phrases": True})
    file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".svg"
    output_path = Path("./" + file_name)
    output_path.open("w", encoding="utf-8").write(svg)
    # Open an existing SVG file
    drawing = Drawing(filename=output_path)

    # # Do something with the drawing, e.g., modify or add elements
    # drawing.add(drawing.rect(insert=("0px", "0px"), size=(
    #     "100%", "100%"), fill=rgb(255, 255, 255)))
    # # Save the modified drawing
    # drawing.saveas(f"w_{output_path}")


def get_subject_phrase(doc):
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]


def get_object_phrase(doc):
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]


def get_dative_phrase(doc):
    for token in doc:
        if ("dative" in token.dep_):
            # a token's subtree refers to the collection of tokens that are directly or indirectly dependent on that token.
            subtree = list(token.subtree)
            # Each token in spaCy has an i attribute that represents its index within the parent Doc object.
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]


def get_prepositional_phrase_objs(doc):
    prep_spans = []
    for token in doc:
        if ("pobj" in token.dep_):
            # a token's subtree refers to the collection of tokens that are directly or indirectly dependent on that token.
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            prep_spans.append(doc[start:end])
    return prep_spans

# A clause is a grammatical unit that contains a subject and a predicate.
# It is a group of words that expresses a complete thought and can function as a sentence or as part of a sentence.


def get_clauses(doc):

    # Find the root token
    root_token = find_root_of_sentence(doc)
    # Find the other verbs
    other_verbs = find_other_verbs(doc, root_token)
    token_spans = []
    # Find the token span for each of the other verbs
    all_verbs = [root_token] + other_verbs
    for other_verb in all_verbs:
        (first_token_index, last_token_index) = get_clause_token_span_for_verb(
            other_verb, doc, all_verbs)
        token_spans.append((first_token_index, last_token_index))
    sentence_clauses = []
    for token_span in token_spans:
        start = token_span[0]
        end = token_span[1]
        if (start < end):
            clause = doc[start:end]
            # if (potential_clause_contains_subj(clause)):
            sentence_clauses.append(clause)
    sentence_clauses = sorted(sentence_clauses, key=lambda tup: tup[0])
    return sentence_clauses
