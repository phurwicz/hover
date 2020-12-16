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
