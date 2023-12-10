from __future__ import annotations

import nox


@nox.session
def format(session):
    session.install("black")
    session.run("black", "code", "tests", "scripts")


@nox.session
def tests(session):
    session.install("-r", "requirements.txt")
    session.run("pytest")
