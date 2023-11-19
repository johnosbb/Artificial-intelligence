import spacy
from pathlib import Path
from spacy import displacy
from spacy.tokens import Span, Doc
import textacy as tx
from spacy.matcher import Matcher
import re
from svgwrite import Drawing, rgb
from spacy.tokens.token import Token  # Import the Token type
import lemminflect
from typing import Union,List  # Import the Union type
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve,auc
import matplotlib.pyplot as plt


verb_patterns_for_verb_phrases = [
    [{"POS": "AUX"}, {"POS": "VERB"}, {"POS": "ADP"}],
    [{"POS": "AUX"}, {"POS": "VERB"}],
    [{"POS": "VERB"}]
]

# Assuming y_true and y_score are your PyTorch tensors
# y_true contains true labels (0 or 1)
# y_score contains predicted scores or probabilities
# True Positive Rate: TPR= True Positives/ (False Negatives + True Positives)
# The True Positive Rate measures the proportion of actual positive instances that are correctly predicted as positive.
# It is also known as Sensitivity or Recall.
# False Positive Rate: # FPR = False Positives/ (False Positives + True Negatives)
# The False Positive Rate measures the proportion of actual negative instances that are incorrectly predicted as positive.

def plot_roc(y_true,y_score):
    # Convert PyTorch tensors to NumPy arrays if required
    # y_true_np = y_true.cpu().numpy()
    # y_score_np = y_score.cpu().numpy()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    #plt.show()
    plt.savefig('roc_plot.svg', format='svg')
    
    
    
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


def show_token(token):
    print(f"Text: {token.text}")
    print(f"Index: {token.i}")
    print(f"Lemma: {token.lemma_}")
    print(f"Part of Speech: {token.pos_}")
    print(f"Tag: {token.tag_}")
    print(f"Dependency Label: {token.dep_}")
    print(f"Is Stop Word: {token.is_stop}")
    print(f"Is Punctuation: {token.is_punct}")
    print(f"Is Space: {token.is_space}")
    # a token's "head" refers to the token's syntactic parent in a dependency parse tree. The head of a token is the word that governs or controls the behavior of the current token in the sentence. It represents the main word or element to which the current token is grammatically related.
    print(f"Head Token: {token.head.text}")
    # Print children of the token
    children = [child.text for child in token.children]
    print(f"Children: {', '.join(children)}")
    # Print ancestors of the token
    ancestors = [ancestor.text for ancestor in token.ancestors]
    print(f"Ancestors: {', '.join(ancestors)}")


def show_sentence_parts(doc):
    print(doc)
    print("{:<12} | {:<6} | {:<8} | {:<8} | {:<8} | {:<24} | {:<20} | {:<10} | {:<12}".format(
        'Text', 'Index', 'POS', "Tag", 'Dep', 'Dep Detail', 'Ancestors', 'Children', 'Token Head'))
    print("--------------------------------------------------------------------------------------------------------------------------------")
    for token in doc:
        # the term "ancestors" refers to the set of nodes that are higher in the parse tree hierarchy and lead to the current token or span of tokens.
        ancestors = ' '.join([t.text for t in token.ancestors])
        # "children" refer to the nodes that are directly dependent on the current token or span of tokens in the parse tree. Children can be thought of as the "child" nodes that are connected to the current node.
        children = ' '.join([t.text for t in token.children])

        print("{:<12} | {:<6} | {:<8} | {:<8} | {:<8} | {:<24} | {:<20} | {:<10} | {:<12} ".format(
            token.text, token.i, token.pos_, token.tag_, token.dep_, spacy.explain(token.dep_), ancestors, children, token.head.text))
        print("--------------------------------------------------------------------------------------------------------------------------------")


def show_sentence_parts_as_md(doc):
    print(doc)
    print("| {:<12} | {:<6} | {:<8} | {:<8} | {:<8} | {:<24} | {:<20} | {:<10} | {:<12}".format(
        'Text', 'Index', 'POS', "Tag", 'Dep', 'Dep Detail', 'Ancestors', 'Children', 'Token Head'))
    print("| ------ | ------ | ---- | ------- | ------- | --------- |  ------- | ------- | ------- |")
    for token in doc:
        # the term "ancestors" refers to the set of nodes that are higher in the parse tree hierarchy and lead to the current token or span of tokens.
        ancestors = ' '.join([t.text for t in token.ancestors])
        # "children" refer to the nodes that are directly dependent on the current token or span of tokens in the parse tree. Children can be thought of as the "child" nodes that are connected to the current node.
        children = ' '.join([t.text for t in token.children])
        print("| {:<12} | {:<6} | {:<8} | {:<8} | {:<8} | {:<24} | {:<20} | {:<10} |  {:<12} |".format(
            token.text, token.i, token.pos_, token.tag_, token.dep_, spacy.explain(token.dep_), ancestors, children, token.head.text))

# extract_ccs_from_token
# ccs = Coordinating Conjunctions
# In linguistic analysis, identifying coordinating conjunctions and understanding their usage is important for parsing and interpreting
# the structure of sentences and clauses.
# Coordinating conjunctions play a role in coordinating or joining elements with similar grammatical functions within a sentence.
# How this function works:
# It starts by checking the part-of-speech (POS) tag of the token. If the POS tag is one of ["NOUN", "PROPN", "ADJ"],
# it considers the token and its children to potentially form a conjoined noun phrase.
# These are typically words like nouns, proper nouns, or adjectives that can be part of a noun phrase.
# It creates an initial list called children, which includes the current token.
# It then iterates through the token's children (dependencies) and filters them based on their dependency labels (dep_).
# It only includes children with dependency labels ["advmod", "amod", "det", "poss", "compound"],
# which are typically modifiers, determiners, possessives, or compound words that can be part of a noun phrase.
# It sorts the children list by their positions (token.i is the position of the token in the document) to ensure they are in the correct order.
# It creates a list called entities to store extracted conjoined noun phrases.
# Initially, it contains a single Span object that spans from the start position of the first child to the end position of the last child.
# Next, it checks if there are any children of the token with the dependency label "conj" (conjunct).
# If such children are found, it recursively calls the extract_ccs_from_token function on those children
# and appends the resulting entities to the current list.
# Finally, it returns the list of extracted conjoined noun phrases (entities).


def extract_ccs_from_token(token: Token )-> Span:
    if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
        children = sorted(
            [token]
            + [
                c
                for c in token.children
                if c.dep_ in ["advmod", "amod", "det", "poss", "compound"] # If we are dealing with a noun or adjective then we are only interested in children with dependency labels ["advmod", "amod", "det", "poss", "compound"] which are typically modifiers, determiners, possessives, or compound words that can be part of a noun phrase.
            ],
            key=lambda x: x.i,
        )
        entities = [Span(token.doc, start=children[0].i,
                         end=children[-1].i + 1)]
    else:
        entities = [Span(token.doc, start=token.i, end=token.i + 1)] # if it is not a noun or adjective, the span is simply the token
    for c in token.children:
        if c.dep_ == "conj": # In spaCy, the "conj" dependency label is used to indicate a conjunct in a coordinated structure, for example "The cat and the Dog"
            entities += extract_ccs_from_token(c) # if a noun has a child with a dependency "conj" then we also need to analyse its conjunct.
    return entities


def extract_ccs_from_token_at_root(span):
    if span is None:
        return []
    else:
        return extract_ccs_from_token(span.root)


def show_noun_chunks(doc):
    print("Noun Chunks\n")
    for noun_chunk in doc.noun_chunks:
        print(f"{noun_chunk.text}  Start: {noun_chunk.start} End: {noun_chunk.end} Root: {noun_chunk.root}")

# this needs to be much more complex, ideally we should pass a subject span


def find_grammatical_person(token):
    if token.pos_ == "PRON" and token.tag_ == "PRP" and token.dep_ == "nsubj":
        if(token.text in ["i", "I"]):
            return "FPS"
        elif(token.text in ["We", "we"]):
            return "FPP"
        elif(token.text in ["You", "you"]):
            return "SPS"
        elif(token.text in ["He", "he", "she", "she", "It", "it"]):
            return "TPS"
        elif(token.text in ["They"]):
            return "TPP"
        else:
            return "TPP"
    elif(token.pos_ == "PROPN" and token.dep_ == "nsubj"):
        return "TPP"
    else:
        return None

# This function is designed to inflect (change the form of) a verb token based on a specified inflection,
# but it applies some conditions to determine whether the inflection should be applied.
# It checks if inflect is True.
# It checks if the part-of-speech (POS) tag of the token is "VERB," indicating that it's a verb.
# It checks that the token is not preceded by an auxiliary verb.
# This is done by examining the POS tags of the token's left children (token.lefts)
# and ensuring that there are no tokens with the POS tag "AUX" in the left children.
# It checks that the dependency label of the token is not "pcomp."
# This last check to exclude cases where the token is part of a prepositional complement (pcomp) structure.
# In English grammar, a prepositional complement is a phrase that follows a preposition and typically provides additional information about the verb.
# verbs within prepositional complements may not need or should not undergo the same inflection.
# Changing the form of a verb within a prepositional complement could result in grammatically incorrect or unnatural sentences.
# By excluding tokens with the dependency label "pcomp," the function is more likely to focus on inflecting main verbs,
# which are the core verbs that convey the primary action or state of the sentence. These are the verbs that are typically
# inflected to match the subject and tense of the sentence
def inflect_token(token, inflect):
    if (
        inflect
        and token.pos_ == "VERB"
        and "AUX" not in [tt.pos_ for tt in token.lefts]
        # t is not preceded by an auxiliary verb (e.g. `the birds were ailing`)
        and token.dep_ != "pcomp"
    ):  # t `dreamed of becoming a dancer`
        return str(token._.inflect(inflect))
    else:
        return str(token)
    
    
def convert_to_past_tense(nlp, doc):
    new_sentence = []
    nominal_subject = None
    for token in doc:
        if token.pos_ == "PRON" and token.tag_ == "PRP" and token.dep_ == "nsubj":
            nominal_subject = token
        elif(token.pos_ == "PROPN" and token.dep_ == "nsubj"):
            nominal_subject = token
        if token.tag_ == "TO" and token.pos_ == "PART":  # the auxiliary part of verbs in infinitive form
            new_sentence.append(token.text)
        elif token.pos_.upper() == "AUX" and token.dep_ == "ROOT":  # if the verb is an auxiliary and a root
            past_form = token._.inflect('VBD')
            if(nominal_subject):
                if(token.text == "are" and find_grammatical_person(nominal_subject) != "FPS"):
                    past_form = "were"
            new_sentence.append(past_form)
        # this is acting as an auxiliary verb to another verb
        elif token.dep_.upper() == "AUX" and token.head.pos_ == "VERB":
            past_form = token._.inflect('VBD') # Verb, past tense.
            if(nominal_subject):
                if(token.text == "are" and find_grammatical_person(nominal_subject) != "FPS"):
                    past_form = "were"
            new_sentence.append(past_form)
        # if the token is acting as a clausal complement to another verb
        elif token.dep_.lower() == "ccomp" and token.head.pos_ == "VERB":
            new_sentence.append(token._.inflect('VBD'))
        # VBG is a gerund, if it is a gerund we should leave it in gerund form
        elif(token.pos_.upper() == "VERB" and token.dep_ == "ROOT" and token.tag_ != "VBG"):
            new_sentence.append(token._.inflect('VBD')) # Verb, past tense.
        else:
            new_sentence.append(token.text)
    new_doc = nlp(" ".join(new_sentence))
    return new_doc


# we must provide a target word (or a target POS) and a replacement


def replace_token(nlp, existing_doc, replacement_word, target_word=None, target_pos=None):
    new_sentence = []
    new_doc = existing_doc
    if(target_pos):
        for token in existing_doc:
            if token.pos_ in target_pos:
                # Replace the verb token with a new value
                new_sentence.append(replacement_word)
            else:
                new_sentence.append(token.text)
        new_doc = nlp(" ".join(new_sentence))
    else:
        if(target_word):
            for token in existing_doc:
                if token.text == target_word:
                    # Replace with a new value
                    new_sentence.append(replacement_word)
                else:
                    new_sentence.append(token.text)
            new_doc = nlp(" ".join(new_sentence))
    # Create a new Doc with the modified tokens
    return new_doc


def show_sentence_structure(doc):

    print("-------------------------")
    for token in doc:
        print(
            f"Token: {token.text} POS: {token.pos_} - [{spacy.explain(token.pos_)}]  Dependencies: {token.dep_} - [{spacy.explain(token.dep_)}] Fine Grained Tag: {token.tag_} - [{spacy.explain(token.tag_)}]")
    for chunk in doc.noun_chunks:
        print(f" Chunk: {chunk.text.lower()}")
    print("-------------------------")



# Find any determiners for the given noun and its associated doc object
# In English, common examples of determiners include "the," "a," "an," "some," "my," "his," "her," "many," and "several."

def find_determiners_for_noun(noun : Token, doc : Union [Doc,Span]) -> Token:
    for child in noun.children:
        if(child.pos_ == "DET"):
            return child
    return None


# Given a child token, find that child's parent token
def find_parent_token_for_child(target_child : Token, doc : Union [Doc,Span]) -> Token:
    for token in doc:
        for child in token.children:
            if child == target_child:
                return token
    return None


# Given a verb span and the associated doc object return the associated subject as a span
def find_subject_in_passive_construction(verb_span : Span, doc : Doc) -> Span:
    number_of_parts = len(verb_span)
    if number_of_parts > 1:
        for part in verb_span:
            print(f"verb part {part.text}")
            if(part.tag_ == "VBN"): # : It checks if the current part of the verb is a past participle verb form (typically used in passive constructions).
                parent = find_parent_token_for_child(part, doc) #  find the parent token of the verb part in the dependency tree. This parent token is often related to the subject of the passive construction.
                if(parent):
                    if parent.dep_ == "nsubj": # It checks if the dependency label of the parent token is "nsubj," which is often used to label subjects in spaCy's dependency parsing.
                        # now we find any determiners
                        determiner = find_determiners_for_noun(parent, doc) # Are there any determiners associated with this noun?
                        if(determiner):
                            parent_as_span = doc[determiner.i: parent.i + 1] # If there are determiners we include those in the span
                        else:
                            parent_as_span = doc[parent.i: parent.i + 1]
                        print(
                            f"verb : {verb_span} has an active voice subject {parent_as_span}")
                        return parent_as_span
    return None



# Given a document and a verb span, find the associated spans that represents the subject
def extract_subjects(verb: Span, doc: Doc) -> Span:
    root = verb.root
    while root:
        if(root.children):
            # Can we find subject at current level by looking for Nominal Subjects or a Passive Nominal Subjects?
            for child in root.children:  # examine the children of the verb
                # is it a Nominal Subject or a Passive Nominal Subject. Note: there are other less common dependencies that can indicate subjects:
                # csubj (Clausal Subject): This label is used to identify clausal subjects, which are entire clauses that function as the subject of the main clause.
                # expl (Expletive Subject): This label is used for expletive subjects, which are placeholders like "it" or "there" that don't have a clear referent.
                # csubjpass (Clausal Subject in Passive): Similar to "csubj," this label is used to identify entire clauses that function as the subject in passive voice sentences.
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = find_span_for_token(child)
                    if(child.dep_ == "nsubj"):
                        print(
                            f"The verb phrase that contains [{verb}] has a child dependency [{child.dep_}] that points to a Nominal Subject: [{subject}].")
                    else:
                        subject_in_passive_voice_construction = find_subject_in_passive_construction(
                            verb, doc)
                        if(subject_in_passive_voice_construction):
                            print(
                                f"The verb phrase that contains [{verb}] has a child dependency [{child.dep_}] that points to a subject in passive voice construction: [{subject}].")
                            return subject_in_passive_voice_construction
                        else:
                            print(
                                f"The verb phrase that contains [{verb}] has a child dependency [{child.dep_}] that points to a Passive Nominal Subject: [{subject}].")
                    return subject
        else:
            print(f"The verb [{verb}] has no children")
        # If we cannot find children which are subjects then we recurse up one level in the sentence tree by looking for dependencies that point towards other clauses
        if (root.dep_ in ["conj", "cc", "advcl", "acl", "ccomp", "pcomp","auxpass"]
                and root != root.head):
            print(
                f"The verb [{verb}] has a dependency [{root.dep_}] that indicates the presence of other clauses.")
            root = root.head
        else:  # we have a verb with no children or one whose children do not have a nominal or passive nominal subject and which does not appear to have other clauses
            root = None

    return None




# returns a matcher object meeting the verb criteria for the given span

def get_verb_matches_for_span(span: Span) -> Matcher:
    verb_matcher = Matcher(span.vocab)
    verb_matcher.add("Auxiliary verb phrase aux-adv-verb", [
        [{"POS": "AUX"}, {"POS": "ADV", "OP": "+"}, {"POS": "VERB"}]])
    verb_matcher.add("Auxiliary verb phrase aux-verb", [
        [{"POS": "AUX"}, {"POS": "VERB"}]])
    verb_matcher.add("Auxiliary verb phrase", [[{"POS": "AUX"}]])
    verb_matcher.add("Verb phrase", [[{"POS": "VERB"}]],)
    return verb_matcher(span)

# Extract a span representation of the verb phrases associated with the given span and return as a list of these verb phrases as Spans
def get_verb_spans(span:  Union [Doc,Span]) -> list:
    matches = get_verb_matches_for_span(span)
    # Filter matches (e.g. do not have both "has walked" and "walked" in verbs)
    verb_spans = []
    for match in [span[start:end] for _, start, end in matches]:
        if match.root not in [vp.root for vp in verb_spans]:
            verb_spans.append(match)
    return verb_spans

# For a given sentence and an associated matches, return the spans associated with those matches
def extract_spans_from_matches(sent : Span , matches : list) -> list:
    spans = []
    for match_id, start, end in matches:
        # Create a span from the match indices
        span = sent[start:end]
        spans.append(span)
    return spans






# Given an entity or token, find the complete span associated with it by finding its children - originally extract_span_from_entity
def find_span_for_token(token : Token) -> Span:
    children = []
    # The .subtree attribute in spaCy includes not only the immediate children of the token but also all descendants,
    # including the children of the children, and so on. It represents the entire subtree of the token in the syntactic parse tree.
    for child in token.subtree:  # The subtree includes the token itself and all the tokens that are dependent on it, directly or indirectly, in the parse tree. This can be especially useful for extracting information associated with a specific word in a sentence or document.
        children.append(child)
    # This will sort the list of children based on the values returned by x.i. In other words, it will sort the children in ascending order of their positions in the document.
    entity_subtree = sorted(children, key=lambda x: x.i)
    extracted_span = Span(
        token.doc, start=entity_subtree[0].i, end=entity_subtree[-1].i + 1)  # The -1 index is used to access the last element in the entity_subtree list, which represents the last token in the sorted subtree.
    return extracted_span


# Finds a matching child for a given span based on a list of allowed types
def find_matching_child_span(root: Token, allowed_types: list)-> Span:
    for token in root.children:
        #print(f"Root: {root} - Child Token : {token} Child Token Dependency : {token.dep_}")
        if token.dep_ in allowed_types:
            return find_span_for_token(token)
    return None


def find_object_as_span_for_token(root: Token) -> Span:
    return find_matching_child_span(root, ["dobj", "dative"])
    
    
def find_compliment_as_span_for_token(root: Token) -> Span:
    complement = find_matching_child_span(
    root, ["ccomp", "acomp", "xcomp", "attr"]    )
    return complement

def check_if_span_is_subspan(span_to_check, existing_spans):
    for span in existing_spans:
        if span == span_to_check:  # we do not check a span against itself
            continue
        start_index_span2 = span.start
        end_index_span2 = span.end
        start_index_span1 = span_to_check.start
        end_index_span1 = span_to_check.end
        if start_index_span1 >= start_index_span2 and end_index_span1 <= end_index_span2:
            return True
    return False


# This will create a unique list of verb phrases (spans). It filters out any redundancy due to the use of auxiliary verbs for example
# if a span has the same root as a previous span we do not add it
# This does not return unique verb phrases in all cases, it return "has" and "has eaten", I assume we only want one of these.
def get_unique_verb_spans(span: Span) -> List[Span]:
    matches = get_verb_matches_for_span(span)
    verb_spans = []
    # we filter them to ensure they are unique
    for match in [span[start:end] for _, start, end in matches]:
        # print(f"Checking span : {match} with root = {match.root}")
        found = False
        for vp in verb_spans:
            # In a Span, the root property refers to the root token of the span, not the entire document. The root token of a span is the token within that span that serves as the head or main word of the span. It might not necessarily be a verb; it depends on the specific syntactic structure of the span.
            if match.root == vp.root:  # ensure they do not have the same root, so do not have both "has walked" and "walked" in the list
                # print(
                #     f"match = {match} root = {match.root} ==  vp = {vp} root = {vp.root}")
                found = True
                break

        if not found:  # we only add them if they do not already exist
            verb_spans.append(match)
    unique = []
    for verb_phrase_span in verb_spans:
        if (check_if_span_is_subspan(verb_phrase_span, verb_spans)):
            continue
        else:
            unique.append(verb_phrase_span)
    return unique  # a unique list of verb chunks (spans) in the form of matches


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
