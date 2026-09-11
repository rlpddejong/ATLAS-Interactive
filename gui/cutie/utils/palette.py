import numpy as np

### SurgeNetSeg Palette ###

color_palette = {
    # Abdomen IDs
    1: (255, 255, 255),  # Tools/camera - White
    2: (0, 0, 255),      # Vein (major) - Blue
    3: (255, 0, 0),      # Artery (major) - Red
    4: (255, 255, 0),    # Nerve (major) - Yellow
    5: (0, 255, 0),      # Small intestine - Green
    6: (0, 200, 100),    # Colon/rectum - Dark Green
    7: (200, 150, 100),  # Abdominal wall - Beige
    8: (250, 150, 100),  # Diaphragm - light Beige
    9: (255, 200, 100),  # Omentum - Light Orange
    10: (180, 0, 0),     # Aorta - Dark Red
    11: (0, 0, 180),     # Vena cava - Dark Blue
    12: (150, 100, 50),  # Liver - Brown
    13: (0, 255, 255),   # Cystic duct - Cyan
    14: (0, 200, 255),   # Gallbladder - Teal
    15: (0, 100, 255),   # Hepatic vein - Light Blue
    16: (255, 150, 50),  # Hepatic ligament - Orange
    17: (255, 220, 200), # Cystic plate - Light Pink
    18: (200, 100, 200), # Stomach - Light Purple
    19: (144, 238, 144), # Ductus choledochus - Light Green
    20: (247, 255, 0),   # Mesenterium
    21: (255, 206, 27),  # Ductus hepaticus - Red
    22: (200, 0, 200),   # Spleen - Purple
    23: (255, 0, 150),   # Uterus - Pink
    24: (255, 100, 200), # Ovary - Light Pink
    25: (200, 100, 255), # Oviduct - Lavender

    # RARP
    26: (150, 0, 100),   # Prostate - Dark Purple
    27: (255, 200, 255), # Urethra - Light Pink
    28: (150, 100, 75),  # Ligated plexus - Brown
    29: (200, 0, 150),   # Seminal vesicles - Magenta
    30: (100, 100, 100), # Catheter - Gray
    31: (255, 150, 255), # Bladder - Light Purple
    32: (100, 200, 255), # Kidney - Light Blue

    # Thorax IDs
    33: (150, 200, 255), # Lung - Light Blue
    34: (0, 150, 255),   # Airway (bronchus/trachea) - Sky Blue
    35: (255, 100, 100), # Esophagus - Salmon
    36: (200, 200, 255), # Pericardium - Pale Blue
    37: (100, 100, 255), # V azygos - Blue
    38: (0, 255, 150),   # Thoracic duct - Green Cyan
    39: (255, 255, 100), # Nerves - Yellow

    # Non-anatomical structures
    40: (150, 150, 150),  # Ureter - Gray
    41: (50, 50, 50),     # Non anatomical structures - Dark Gray
    42: (0, 0, 0),        # Excluded frames - Black

    # Additional structures
    43: (173, 216, 230), # Mesocolon - Light Blue 2
    44: (255, 140, 0),   # Adrenal Gland
    45: (223, 3, 252), # Pancreas (252, 186, 3)
    46: (0, 80, 100), # Duodenum
}

custom_names = {
    1: "Tools/camera",
    2: "Vein (major)",
    3: "Artery (major)",
    4: "Nerve (major)",
    5: "Small intestine",
    6: "Colon/rectum",
    7: "Abdominal wall",
    8: "Diaphragm",
    9: "Omentum",
    10: "Aorta",
    11: "Vena cava",
    12: "Liver",
    13: "Cystic duct",
    14: "Gallbladder",
    15: "Hepatic vein",
    16: "Hepatic ligament",
    17: "Cystic plate",
    18: "Stomach",
    19: "Ductus choledochus",
    20: "Mesenterium",
    21: "Ductus hepaticus",
    22: "Spleen",
    23: "Uterus",
    24: "Ovary",
    25: "Oviduct",
    26: "Prostate",
    27: "Urethra",
    28: "Ligated plexus",
    29: "Seminal vesicles",
    30: "Catheter",
    31: "Bladder",
    32: "Kidney",
    33: "Lung",
    34: "Airway (bronchus/trachea)",
    35: "Esophagus",
    36: "Pericardium",
    37: "V azygos",
    38: "Thoracic duct",
    39: "Nerves",
    40: "Ureter",
    41: "Non anatomical structures",
    42: "Excluded frames",
    43: "Mesocolon",
    44: "Adrenal Gland",
    45: "Pancreas",
    46: "Duodenum",
}


# custom_names = {
#     1:  'Hook',
#     2:  'Forceps',
#     3:  'Suction & irrigation',
#     4:  'Vessel sealer',
#     5:  'Right lung',
#     6:  'Azygos vein',
#     7:  'Vena cava',
#     8:  'Aorta',
#     9:  'Pericardium',
#     10: 'Airways',
#     11: 'Vagal nerves',
#     12: 'Recurrent laryngeal nerves',
#     13: 'Thoracic duct incl. fat',
#     14: 'Thoracic duct',
#     15: 'Left pleura',
#     16: 'Esophagus',
#     17: 'Clip applier',
#     18: 'Needle driver',
#     19: 'Scissors',
#     20: 'Stapler',
# }

# color_palette  = {
#     1:  (100,  80,   0),            # Hook                          (Orange)
#     2:  (128,   0,   0),            # Forceps                       (Red)
#     3:  (  0, 128,   0),            # Suction & irrigation          (Green)
#     4:  (  0, 255, 255),            # Vessel sealer                 (Cyan)
#     5:  (160, 100, 160),            # Right lung                    (Pink)
#     6:  (  0,   0, 255),            # Azygos vein                   (Blue)   
#     7:  (  0,   0, 100),            # Vena cava                     (Dark Blue)  
#     8:  (255,   0,   0),            # Aorta                         (Red)
#     9:  (255,   0, 157),            # Pericardium                   (Purple)
#     10: (255, 255, 255),            # Airways                       (White)
#     11: (255, 255,   0),            # Vagal nerves                  (Yellow)
#     12: (139, 128,   0),            # Recurrent laryngeal nerves    (Dark Yellow)
#     13: (  0, 255,   0),            # Thoracic duct incl. fat       (Green)
#     14: (  0, 100,   0),            # Thoracic duct                 (Dark Green)
#     15: (140,  80, 180),            # Left pleura                   (Violet)
#     16: (255, 160,   0),            # Esophagus                     (Orange)
#     17: (  0, 128, 128),            # Clip applier                  (Teal)
#     18: (200, 255, 100),            # Needle driver                 (Lime-Yellow Green)
#     19: (180, 180, 255),            # Scissors                      (Pastel Blue)
#     20: (128, 100, 160),            # Stapler                       (Muted Lavender)
# }

# color_palette = {
#     1: (255, 0, 0), # Class 1, red
#     2: (0, 0, 255), # Class 2, blue
#     3: (0, 255, 0), # Class 2, green
# }

# custom_names = {
#     1: "Class 1",
#     2: "Class 2",
#     3: "Class 3",
# }

custom_palette_np = np.array([color_palette.get(i, (0, 0, 0)) for i in range(len(custom_names)+1)])
custom_palette = custom_palette_np.astype(np.uint8).tobytes()
