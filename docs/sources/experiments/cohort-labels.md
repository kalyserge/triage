# Cohort and Label Deep Dive

## What are Cohorts and Labels in Triage?

This document assumes that the reader is familiar with the concept of a machine learning target variable and will focus on explaining what is unique to Triage.

**A label is the binary target variable for an entity at a given as-of-date and a given label timespan, independent of any cohort**

**A cohort is the list of entities that are used for modeling on a given as-of-date, independent of any labels**

These might be dense statements, so let's unpack a few things.

- An entity refers to whatever or whomever is the object of prediction. For instance, facilities to inspect. Triage expects the entity ids to be integers.
- Cohorts and labels are both based on an `as-of-date`, as opposed to being based on specific matrices.
- Triage is concerned with events, so labels also consider a `label timespan` as part of their definition: Two label definitions with the same `as-of-date` but different label timespans will contain different collections of events, and therefore may have different values.


Both labels and cohorts are defined in Triage's experiment configuration using SQL queries, with the variables (`as_of_date`, `label_timespan`) given as placeholders. This allows a wide variety of label and cohort definitions.


## Cohort Simple Example

Let's say I am prioritizing the inspection of restaurants. One simple definition of a cohort for restaurant inspection would be to include *any restaurants that have active permits in the last year* in the cohort. Assume that these permits are contained in a table, named `permits`, with the facility's id, a start date, and an end date of the permit.


entity_id | start_date | end_date
------------ | ------------- | ------------
25 | 2016-01-01  | 2016-02-01
44 | 2016-01-01  | 2016-02-01
25 | 2016-02-01  | 2016-03-01

Triage expects the cohort query passed to it to return `entity_id`s given an `as_of_date`, and it runs the query for each `as_of_date` that is produced by your temporal config. You tell Triage where to place each `as_of_date` with a placeholder surrounded by brackets: `{as_of_date}`. An example query that implements the 'past year' definition would be:

`select distinct(entity_id) from permits where tsrange(start_date, end_date, '[]') @> {as_of_date}`

- Running this query using the `as_of_date` '2017-01-15' would return both entity ids 25 and 44.
- Running it with '2017-02-15' would return only entity id 25.
- Running it with '2017-03-15' would return no rows.

The way this looks in an Experiment configuration YAML is as follows:

```
cohort_config:
  query: |
    select entity_id
    from permits
    where
    tsrange(start_time, end_time, '[]') @> {as_of_date}
  name: 'permits_in_last_year'
```

The `name` key is optional. Part of its purpose is to help you organize different cohorts in your configuration, but it is also included in each matrix's metadata file to help you keep them straight afterwards.

## Label Simple Example

...



### include_missing_labels_in_train_as

The label configuration has a little boolean flag that can drastically change the structure of your experiment. `include_missing_labels_in_train_as` is concerned with all of the rows that show up in the cohort for a given date that *do not* show up in the label query. How should they be represented in the prediction matrices that include this `as_of_date`?

- If you omit the flag, they show up as missing. This is common for inspections problems, wherein you really don't know a suitable label. The facility wasn't inspected, so you really don't know what the label is. This makes evaluation a bit more complicated, as some of the facilities with high risk scores may have no labels. But this is a common tradeoff in inspections problems.
- If you set it to True, that means that all of the rows have positive label. What does this mean? It depends on what exactly your label query is, but a common use would be to model early warning problems of dropouts, in which the *absence* of an event (e.g. a school enrollment event) is the positive label.
- If you set it to False, that means that all of these rows have a negative label. A common use for this would be in early warning problems of adverse events, in which the *presence* of an event (e.g. excessive use of force by a police officer) is the positive label.
