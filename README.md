# TR7AE-Mesh-Editor
Blender addon for import/export of Tomb Raider Legend/Anniversary models

![image](https://github.com/user-attachments/assets/e7ab59bf-9154-4941-a179-57b487745039)

In comparision to the Noesis plugin I previously made 3 years ago, this Blender addon is full of brand new features, available thanks to a deeper understanding of the format and thanks to Blender's API which allows to import much more metadata than Noesis allows.

## Features include

- HInfo import/export such as HBox, HSphere, HMarkers and HCapsules
- HInfo live editing such as position, rotation and scale (where applicable) and export
- Export is no longer limited to character models only/will no longer cause issues on scene objects/props/vehicles and such
- Vertex color import/export
- Environment Mapping and Eye Reflection Environment Mapping import/export, to create meshes with reflection
- File size of exported models is now significantly reduced thanks to a more optimized way of storing weights
- Material and texture import, including live editing Blending mode
- Target and Markup import/ Target export
- Adding HInfo Components to bones
- PCD > Image, Image > PCD conversion
- RAW > Image, Image > RAW conversion

(Example of vertex colors import and export)
![image](https://github.com/user-attachments/assets/d091a715-27c8-499d-a7cd-52276eb7cd7a)
![image](https://github.com/user-attachments/assets/962cdcf0-aed4-40ee-8735-fb1383037244)

The addon also allows to snap an Armature's first bone's position and rotation directly to an HMarker. This allows live editing of things like weapon attachment. No more blindly changing float values to achieve the result you want.

![image](https://github.com/user-attachments/assets/219cbc91-38e2-4a32-9ba1-d064bb372866)
![image](https://github.com/user-attachments/assets/16e04bb3-9fc0-4d95-8b37-37c2e3eebc45)

The addon is currently capable of importing cloth physics data, however nothing is done with that during export. Cloth physics data still has to be properly implemented into the addon. In the future, it will be possible to export cloth data and create new physics bones.

## What's planned

- Write vertices from face corners instead of actual vertex data. This will fix issues such as UV bleeding when exporting a model with removed double vertices
- Write Markups
- Cloth export
- Level Import/Export
- Better handling of MFace data

## Known limits

- The limit of vertex count for one entire model is 21,845
- If any material has the "Flat Shading" flag enabled, the limit of Virtual Segments (bone weighting entries) is 153

## Credits

* [DKDave](https://github.com/DKDave) for being the OG creator of the Noesis plugin's importer, of which much logic has been transferred to the Blender addon
* [TheIndra55](https://github.com/TheIndra55) for his amazing [Menu Hook](https://github.com/TheIndra55/TRLAU-menu-hook) which made debugging fast, easy and of course mandatory for installing mods
* [arcusmaximus](https://github.com/arcusmaximus) for general help and suggestions
* [Joschka](https://forum.xentax.com/memberlist.php?mode=viewprofile&u=82197) for originally helping with the Noesis plugin, of which much logic has been transferred to the Blender addon
* [HenrysArts](https://x.com/HenrysArts) for helpful testing, debugging, suggestions and inputs


### For help and support, join the Tomb Raider Modding Discord: https://discord.gg/euRUvFkpjb
