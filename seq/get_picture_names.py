import os

# Pfad zum Ordner mit den JPG-Dateien
basis_pfad = os.path.dirname(__file__)  # Pfad zu lala.py
ordnerpfad = os.path.join(basis_pfad, 'seq', 'sichere_schlagloecher')

# Leeres Dictionary zum Speichern der Dateinamen
bilder_dict = {}

# Alle Dateien im Ordner durchgehen
for datei in os.listdir(ordnerpfad):
    if datei.lower().endswith('.jpg'):
        dateiname_ohne_endung = os.path.splitext(datei)[0]
        print(f'Gefunden: {dateiname_ohne_endung}')
        bilder_dict[dateiname_ohne_endung] = None  # oder ein anderer Initialwert

# Optional: das Dictionary anzeigen
print("\nErstelltes Dictionary:")
print(bilder_dict)
