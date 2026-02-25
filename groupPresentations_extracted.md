# groupPresentations.pdf — Extracted Text

> Auto-extracted from the uploaded PDF. Page sections preserved.

## Page 1

Text Mining for Economics & Finance
Group Presentation Guidelines
Group Formation
Each group should have 4-5 members (in very rare cases I will allow groups of 3 or 6,
however you should ask me for permission first). You should already be discussing potential
projects with your classmates. Try to find a group with similar interests and a mix of skills. For
those of you who are more experienced, you will learn more if you are teaching/leading others
in your group.
Final groups will be determined by February 6. When you have a group together, please
submit a proposal to the class email. You should have a list of group members, and a dataset
you are interested in. You should also have a research question in mind (this is non-binding, but
I'd like you to have a proposed investigation in mind). For the proposal stage, it's fine if you do
not have a full group (i.e. only 1, 2, or 3 people). When you have this ready, please email me at
textminingimperial@gmail.com.
During the second half of class on the 29th, I will ask everyone who submitted a
proposal to give a short (less than a minute) description of who is in their group, and what they
plan on studying. If you cannot attend class at that time (e.g. you are watching the recordings) I
will read your submitted proposal out loud to the class on your behalf.
This for three reasons - first, I want to vet all the ideas to make sure your ideas are at
the right level of ambition (i.e. not too hard, not too easy). Second, I want you all to share your
ideas with the group, so you can see what each other are working on. This will improve the
overall quality of everyone's final presentations. Third, in case there are students who do not
have a group yet, I want to give them a chance to find a group working on similar interests.
Final proposals are due by the end of that week (Feb 1). I will approve all proposals by
Monday Feb 12 (or else ask for a revision to the idea, which should be resubmitted for rapid
approval that week). If there are issues getting everyone in a group (some people without a
group, some groups too big/small) I will re-assign people to make sure there is balance. I hope
this is unnecessary, as I expect you all to be friendly and eager to learn from one another.
Recommended Workflow
Once you have chosen a group, a dataset, and a research question, you should start
working on your analyses. Note that many datasets (including those provided by me) are large
and slow to process. I would recommend doing a lot of exploring with a small subset, and
once you think you have a good idea, then scale up and run the analyses on a larger pool.
Early on, produce lots of graphs - of text, but especially the metadata. Make sure you
know what structures in the world are being revealed in your text. Your presentation should
include the most insightful tables and graphs from your explorations, but not all of them.
Make sure your research question is interesting enough. Don't test questions that are
too easy (e.g. is this a yelp review of a restaurant or a hair salon) or too hard (is this yelp review
written by a left-handed or right-handed person). Ideally your question should require you to
rely on multiple bits of metadata. For example: do women use different language to for high-
star ratings than men do? Don't forget that you can use the metadata to focus models on
subsets of the data, as well as targets for prediction models. If your question is challenging

---

## Page 2

enough (e.g. if you merge in metadata from other sources) then you may be successful with a
simpler question. If you have doubts about this, please ask me for clarification as early as
possible, so I can provide guidance.
Presentation Format
You will have 10 min for presentation, and 3 min for Q&A. Everyone in the group should
e expected to speak. Please prepare a pdf version of your slide deck for easy transitions during
class. Everyone should submit their slide decks and R code before class on March 5, to
textminingimperial@gmail.com. I will then determine the presentation order randomly, to be
fair to everyone - though if your group requires an early or late presentation time, for example if
some of you are in a different time zone, then please make that request as soon as possible,
and I may be able to accommodate.
Although there is room for variations, most of the best presentations should follow a similar
story arc. First, motivate your research question - why do you care about what you're
studying? If you did learn something interesting, what would be the impact? State your
quantities of interest and populations of interest. Second, show some descriptive work - how
many documents do you have, how long are they, and what are the distributions of the
metadata you care about? Which subset of the data do you plan to focus on, and why (e.g.
incomplete records, outlier cases, etc.)? Third, apply some NLP tools. Describe what
techniques you used, and why you thought they were appropriate for your research question.
Ideally, you should draw from at least three different lectures, using a combination of
techniques to shed light on your data. These analyses should produce insight - interpret your
data to show your audience what you learned about your domain. Fourth, build one or two
prediction models, and evaluate their accuracy. Choose a sensible procedure for accuracy
evaluation. You must compare your results to some simple benchmark models, to show the
added value of your more complex analyses. Fifth, discuss how your analyses could be
improved, and the limitations of your approach. When do you think it will be most successful,
and when might it fail?
Marking Scheme
My TAs and I will grade every presentation on four categories, each of which will be
given equal weight.
Analysis - did you choose and execute the right methods for your research question?
Presentation - did you explain your methods and results clearly to an outside audience
Q&A - can you discuss the boundary conditions of your model, and next steps?
Code - did you keep a well-documented record of your analyses?
Alternative Essay
If you have a unique circumstance that will prevent you from completing a group project
and presentation effectively, please contact me ASAP. I will allow some students with valid
excuses (subject to approval by the finance program team) to complete an essay project
instead. This will involve all the same work, but will be written instead of delivered in person.
Essays should also be accompanied by analysis code, include several graphs and figures, and
so on. Instead of a question and answer session, I expect more. The essay should be between
1,500 and 2,000 words, not including figures, references, and tables. My expectation is that
very few students in the class will need this option. However, I recognize we all are dealing with
unique circumstances and if travel, health, etc. intervene this option will allow you to still fulfill
the requirements of the class.

---

## Page 3

Datasets
Here are some options you can pick from. I have highlighted five datasets that I have
used to prepare your in-class assignments. The assignment data will be too small for you to do
a lot of creative analyses, but the full versions are very large. If you're interested in the full
versions, you can download them from this link. I am happy to answer any questions you might
have about the structure of the data, or what you might want to test using it.
https://drive.google.com/drive/folders/1xkbavFKDqDvwdgHKbxqLnjbB9G9v1_k1?usp=sharing
You are welcome to use your own data. You could start working on a completely
different project (e.g. predict stock returns using news articles or social media), Or you could
find new metadata to merge into one of the existing sets (e.g. if you found neighborhood-level
income data, you could merge that with Yelp data to predict poverty using business reviews).
Extra work like this will be rewarded, if you executer the project well. Note - if you use your
own data, you do not have to submit the code you used to scrape/acquire/merge it. You will
only need to submit the analysis that operates on the cleaned version of the data (i.e. as clean
as the example data below).
Yelp Academic dataset
35,086 restaurants
584,137 reviews
Metadata:
all yelp business information (price, category, parking, alcohol, etc)
city, zip code (for linking to income data, political data, etc)
user ratings of helpful, funny, cool
Glassdoor reviews
97,865 company reviews
Split between large tech companies and anonymous startups
Metadata:
company name (large companies only)
location, date, job title
reviewer ratings - six categories (management, benefits, culture, etc.)
user ratings of review helpfulness
CFPB complaints
2,396,033 consumer complaints about financial institutions
Submitted to CFPB from 2011-2021
Metadata:
company name, location
date, product type, submission format
was the complain resolved, disputed
final resolution, timeliness
how was the complaint submitted

---

## Page 4

Job descriptions
244,768 job descriptions from across the UK posted on Adzuna
Metadata:
starting salary
location
job title, company name
job category
Earnings Calls
13,122 earnings calls from 2010-2013
Text : initial speech plus question & answer period (819,540 turns)
Metadata:
company IBES code (for linking) & FY
actual and expected earnings of quarter
questioner names & company (could link data, determine gender, etc.)
If you want to explore more, here are some other data sources online... I am also open to
projects focusing on other text data, though be careful - there can be a lot of boring data
cleaning that goes into preparing a dataset for text analysis, so make sure you don't choose a
dataset that will be too hard to prepare.
Amazon review data (many categories)
http://jmcauley.ucsd.edu/data/amazon/
Yelp academic dataset (many other business types)
yelp.com/dataset
10K filings, corporate ethics statements, etc.
https://sraf.nd.edu/data/
Upworthy archive - predict success of viral media based on headlines
https://upworthy.natematias.com/

---
