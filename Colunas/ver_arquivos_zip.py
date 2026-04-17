import zipfile

# Ver arquivos no SIM.zip
with zipfile.ZipFile("../Dados_Zip/SIM.zip", 'r') as zf:
    print("📦 Arquivos no SIM.zip:")
    print(zf.namelist())

# Ver arquivos no SINASC.zip
with zipfile.ZipFile("../Dados_Zip/SINASC.zip", 'r') as zf2:
    print("\n📦 Arquivos no SINASC.zip:")
    print(zf2.namelist())
