import os

# ~/.streamlit ディレクトリを作成
streamlit_config_dir = os.path.expanduser("~/.streamlit")

os.makedirs(streamlit_config_dir, exist_ok=True)
config_path = os.path.join(streamlit_config_dir, "config.toml")
print(config_path)

config_content = """
[server]
port = 8501
enableCORS = false
enableWebsocketCompression = true
runOnSave = true

[theme]
# base="light"
# backgroundColor = "#e7eaf2"
# secondaryBackgroundColor = "#afb7c2"
# textColor = "#645b7f"
primaryColor = "#645b7f"
backgroundColor = "#e7eaf2"
secondaryBackgroundColor = "#E0E0E0"
textColor = "#000000"
font = "monospace"

"""

# ファイルに書き込む
with open(config_path, "w", encoding="utf-8") as config_file:
    config_file.write(config_content)

print(f"Streamlit configuration file created at: {config_path}")