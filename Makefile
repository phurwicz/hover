TEST_MODULE_PATH=./hover

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
coverage-report:
	@coverage xml -o cobertura.xml
	@curl https://coverage.codacy.com/get.sh -o get.sh
	@bash get.sh report
