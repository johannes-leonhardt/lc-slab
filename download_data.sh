wget https://uni-bonn.sciebo.de/s/YAefFFLkdMdNQts/download
unzip download 
rm download
find . -name "*.zip" -exec sh -c 'for f; do unzip -d "$(dirname "$f")" "$f" && rm "$f"; done' _ {} +