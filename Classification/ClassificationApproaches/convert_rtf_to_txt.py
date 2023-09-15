
from os import listdir
from os.path import isfile, join
import aspose.words as aw
import os


def rtf_to_text(fileName, outputFile):
    output = aw.Document()
    # Remove all content from the destination document before appending.
    output.remove_all_children()
    input = aw.Document(fileName)
    # Append the source document to the end of the destination document.
    output.append_document(
        input, aw.ImportFormatMode.KEEP_SOURCE_FORMATTING)
    output.save(f"{outputFile}")


output_file_path = './data/Gutenberg/textfiles'
rtf_file_path = './data/Gutenberg/'
rtf_files = [join(rtf_file_path, f) for f in listdir(
    rtf_file_path) if isfile(join(rtf_file_path, f)) and ".rtf" in f]
for rtf_file in rtf_files:
    outfilename = os.path.basename(rtf_file)
    rtf_to_text(rtf_file, f"{output_file_path}/{outfilename}.txt")
    # Provide the paths to the RTF and output text files
