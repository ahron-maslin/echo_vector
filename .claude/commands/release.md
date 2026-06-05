Run the full release pipeline: lint, typecheck, test, commit, tag, and push to trigger CI/CD publish to PyPI.

Usage: /release <version>   (e.g. /release v0.1.3)

Steps:
1. Verify a version argument was provided (format: vX.Y.Z)
2. Run `uv run ruff format --check echovector/ tests/` and `uv run ruff check echovector/ tests/`
3. Run `uv run mypy echovector/`
4. Run `uv run pytest tests/ -m "not slow and not gpu" -q`
5. If there are uncommitted changes, stage and commit them with message "release: $ARGUMENTS"
6. Create git tag `$ARGUMENTS`
7. Push the branch and the tag: `git push origin master` and `git push origin $ARGUMENTS`
8. Report the tag that was pushed and remind the user to monitor CI at https://github.com/ahron-maslin/echo_vector/actions

If any step fails, stop and report the failure before proceeding.
