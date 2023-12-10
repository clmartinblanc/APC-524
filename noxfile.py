from __future__ import annotations

import nox


@nox.session
def tests(session):
    session.install("pytest>=7.0.0", "uncertainties")
    session.run("pytest")


@nox.session
def format(session):
    session.install("black")
    session.run("black", "src", "tests")