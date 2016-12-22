# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:01:44 2015

@author: jensv
"""

from git import Repo
import os


def call_and_git_commit(call='', call_path=None):
    """
    Returns a program call and git HEAD commit hash of a git repo located in
    the supplied call directory. If no directory is passed the current
    directory is used.
    """
    if call_path is None:
        call_path = os.getcwd()
    repo = Repo(path=call_path, search_parent_directories=True)
    commit = repo.commit('HEAD')
    git_commit = commit.hexsha
    return call, git_commit
