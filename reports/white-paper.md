{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Predicting student's failure to meet passing criteria in online Artificial Intelligence course\n",
    "This project aims at predicting the outcome of a student in an online course. The course is run by a popular North American Univarsity, and it uses Canvas as the Learning Management System (LMS). The course has a duration of 9 weeks from stat to finish, we aim at predicting if a student will complete the course after collecting the first 4 weeks of student activity data. This is relevant because if we can obtain advance warning that some students are not on track to complete, the learning facilitators can reach out to the student to provide help, allow for extra time to complete the deliverables or offer to shift the enrollment to a later cohort. \n",
    "\n",
    "## Data\n",
    "\n",
    "The data for this project is real data from 5 cohorts of students in the same 9-week online program. The data is anonymous, individual students are identified by the `student_id` number. This id is meaningless to other students or to the general public. To access the data we make use of the Canvas LMS API. Please note, while every user can make API calls with thier individual API key, a student account will not grant you access to download course level data. You will need to have TA, Instructor, Observer, or Admin privileges, to execute most of these API calls. \n",
    "\n",
    "### Outcome Variable\n",
    "Before any other data extraction effort, we had to identify the outcome variable. This would indicate, wether the student passed `1` or failed `0` the course. In order to obtian this, we submit a GET request to `/api/v1/courses/{COURSE_ID}/students/submissions?student_ids[]={STUDENT_ID}&per_page=200`.\n",
    "This returns the submissions for the relative student in the course. Out of all the submissions, ony the Assignemnts and the Capstone roject count towards the final grade. Counting the number of assignents submitted and graded complete, and comparing it to the passing critieria of 9 out 11 Assignemtns + 1 Capstone, we can create the binary outcome variable.\n",
    "\n",
    "### Feature Engineering\n",
    "While there are many possible data sources, not many contain a timestamp that we can use to pretend we are halfway in the course. For example the same endpoint used for the Outcome variable has informaiton about the submission completion and tardiness, which would be related to the student outcome. however this data represents summary data at the end of the course (because the course has alredy taken pace). For future course, we can imagine collecting this student summary on a daily basis to generate a tomeline of the event.\n",
    "The endpoint which does have a timestamp and that we use for our initial models is `/api/v1/courses/{COURSE_ID}/analytics/users/{STUDENT_ID}/activity?per_page=200`\n",
    "The data pulled with this request contains essentially a count of how many pages have been viewed by the student in any given hour, and the number of submissions for each day. From these two data series we extract and calcuate the major statistical quantities that could be used to augment the prediction.\n",
    "\n",
    "The list of features is:\n",
    "```\n",
    "'student_id'\n",
    "'course_id'\n",
    "'tot_page_views'\n",
    "'average_daily_views'\n",
    "'median_daily_views'\n",
    "'max_daily_views'\n",
    "'days_with_views'\n",
    "'hours_with_views'\n",
    "'max_views_per_hour'\n",
    "'avg_views_per_hour'\n",
    "'median_views_per_hour'\n",
    "'avg_hours_with_views_per_day'\n",
    "'max_hours_with_views_per_day'\n",
    "'median_hours_with_views_per_day'\n",
    "'tot_participations'\n",
    "'average_daily_participations' \n",
    "'median_daily_participations'\n",
    "'max_daily_participation'\n",
    "'days_with_participations'\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exploratory Data Analysis\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:student-fail-prediction]",
   "language": "python",
   "name": "conda-env-student-fail-prediction-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}