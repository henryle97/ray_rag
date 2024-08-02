find . -name "*.py" \
 | xargs python -m isort
find . -name "*.py" \
 | xargs python -m black -l 79
