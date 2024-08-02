export EFS_DIR=/mnt/hdd4T/henryle/llm/dataset
wget -e robots=off --recursive --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.ray.io --no-parent --accept=html \
  -P $EFS_DIR https://docs.ray.io/en/master/
