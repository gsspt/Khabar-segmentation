import json

with open('data/processed/Kitab_Uqala_al_Majanin_annotated.json') as f:
    data = json.load(f)

no_isnad_count = 0
for akhbar in data['akhbar']:
    transmetteurs = akhbar.get('transmetteurs', [])
    # If no transmitters, likely no isnad
    if not transmetteurs or len(transmetteurs) == 0:
        no_isnad_count += 1

print(f"Khabars without transmitters: {no_isnad_count}")
print(f"Percentage: {100*no_isnad_count/len(data['akhbar']):.1f}%")