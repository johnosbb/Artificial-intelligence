#!/usr/bin/env python3

from jinja2 import Template
import os

def generate_markdown_from_template(template_file, output_file, context):
    # Step 1: Read the template from the file
    with open(template_file, 'r') as file:
        template_content = file.read()

    # Step 2: Create a Jinja2 Template object
    template = Template(template_content)

    # Step 3: Render the template with the context (dictionary)
    rendered_content = template.render(context)

    # Step 4: Write the rendered content to the output file
    with open(output_file, 'w') as output:
        output.write(rendered_content)

    print(f"Markdown document has been generated and saved to {output_file}")


# Example usage:

# Define the context (dictionary with dynamic content)
context = {
    'title': 'Automated Report',
    'intro_text': 'This is the introduction of the report.',
    'details': 'Here are the details of the report, including key findings and analysis.',
    'conclusion': 'This is the conclusion with final remarks and recommendations.',
}

# Path to the template file (a .md file)
template_file = '/mnt/500GB/templates/template.md'

# Path for the output file (where the generated Markdown will be saved)
output_file = '/mnt/500GB/templates/generated_report.md'

# Generate the markdown document
generate_markdown_from_template(template_file, output_file, context)
