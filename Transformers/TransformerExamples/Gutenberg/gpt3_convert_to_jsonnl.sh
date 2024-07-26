#!/bin/bash
(cd ./data; openai tools fine_tunes.prepare_data -f ".kant_prompts_and_completions.json")