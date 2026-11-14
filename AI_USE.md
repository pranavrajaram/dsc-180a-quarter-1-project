# AI use and development process

Here are the prompts we used to support our project

> using the handoff doc in the folder that covers the goal of experimenting on tone and LLM performance, can you do some work. and get a preliminary experiment up with clear easy to report results. keep code VERY clean and reproduceable, COMPLETELY AVOID creating millions of slop files/mds keep everything as concise file wise as possible, keeping reproduceability in mind. 
- This was our initial setup prompt after deciding on an idea and writing up a plan of attack.

> please continue the work based on the project and report guidelines linked above, specifically explore adding a "wrong hint" condition to the expeirment to test LLM sycophancy
- This was our prompt to continue analysis after the initial experiment.

> this is pretty useless you didnt really do much besides bookkeeping "validation" and "audit" nonsense. that is completely unehlpful, please actually extend/improve the project as outlined in the doc.
- This was a directed follow up to do a better job of the new analysis. After the agent struggled with the initial prompt, we wrote up a more detailed spec to better guide the analysis.

> can yuou folder up the repo a bit better instead of everything being separate files in main. also update the readme so its very easy to follow for a grader that hasnt read the rpoject yet - they dont care about regenerating latex report or its latex source, just recreating the resultd that drive the report is the omportant part.
- This was to clean up the repo for readability and align with submission requirements.

> please convert this google doc to LaTeX.
- Improving the quality of our writing in the written report

> Here are bulletpoints and a title for several paragraphs of our written report. For each of these paragraphs, turn the bulletpoints and main ideas into a coherent paragraph that flows well from the one before it.
- Getting our resulting data into tables in the report that are easy to read

> Here is the data from our results about model performance across different tones and question types. The math questions had two repetitions, one with a false hint and one with no hint, and the coding questions had two repetitions that were the same. Format this data into a table that will be easy to read in pdf format in our report.
- Converting our written report into LaTeX for submission.

## Two-turn challenge extension (November 14, 2026)

> ok what do you thinkt he best next steps to make our study better owuld be, specificaly on the sycophancy side

> ok now go to work on the two-turn challenge study. be sure to follow the guideliens of reproduceable code and clear communication of results so its easy to interpret

The assistant proposed the matched two-turn design, implemented the collector and
analysis, and ran the local experiments. The author selected this direction and
requested reproducible code and interpretable results. The study continues 80
saved first answers under three follow-up conditions. Tests verify shared parent
conversations and seeds, answer transition categories, conditional denominators,
and rejection of incomplete or altered records. Results are derived from saved
responses and task-cluster bootstrap comparisons, with commands in the README.

The assistant also inspected unparsed and conflicting part-(a) responses and
recorded twelve response-specific annotations in `results/challenge/review.csv`.
These are AI-assisted review decisions, not independent human grading. Both
automated and reviewed states remain available; the README explains the effect
on the primary result. The author can inspect each annotation against the saved
response.
