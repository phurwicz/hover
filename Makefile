TEST_MODULE_PATH=./hover
VULTURE_EXCLUDE=*ipynb_checkpoints*

clean:
	@echo "Cleaning package build files.."
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@echo "Done."
publish:
	@echo "Publishing to PyPI.."
	@python setup.py sdist bdist_wheel
	@twine check dist/*
	@twine upload dist/*
	@echo "Done."
coverage:
	@coverage run --source=$(TEST_MODULE_PATH) -m pytest
	@coverage report -m
vulture:
	@vulture $(TEST_MODULE_PATH) --exclude $(VULTURE_EXCLUDE)
