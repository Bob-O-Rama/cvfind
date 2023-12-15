# Image Tile Library
One issue with testing cvfind is having test cases / problematic tile sets to work with.   A Tile Image Library has been created in an AWS bucket to house zipped up collections of image tiles for use in testing.  These tile sets include both as shot / as stacked image files from actual imaging projects as well as "simulated" dissections of large images. 

# How To Contribute
If you have a challenging stitching project big or small, even if its just some pairs of images that won't alaign, I can host them for others to use in their own testing of stitching software.  This allows developers of any stitching system try their own software against known problem images.  

# Library Contents:

**Note:** the AWS bucket being used to host the zip files currently only supports HTTP, so you may get insipid security warning when downloading the zip files below.

---

**NEC uP7720 DSP Die Photo - 20,000 x 20,000 px** - Dissected into 20 x 20 or 40 x 40 tiles, overlap ~30%.
Fairly challenging owing to lots of repeated, similar structures.  ~300MB per collection, JPG.

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/312adbdb-48e0-49ad-a816-ca3094598d9b)

http://image-tiles.s3-website-us-east-1.amazonaws.com/bobmahar/dissections/dissection_series_a_20_20.zip
http://image-tiles.s3-website-us-east-1.amazonaws.com/bobmahar/dissections/dissection_series_b_40_40.zip

---

**NEC uP7720 DSP Die Photo - Original Source Images** - 240 source images, each about 130MB, includes a duplicated row as shot, ~30% overlap. The duplicate row is not file copies, but rather a row photographed twice, so the image files are distinct, but highly overlapped.  27.8GB for this collection, TIFF.

http://image-tiles.s3-website-us-east-1.amazonaws.com/bobmahar/source/uP7720_PMax.zip

---

**HP Print Head Wafer Scan - 30,000 x 30,000 px** - Dissected into 20 x 20, 40 x 40, and 20 x 30, overlap ~30%.
Extremely challenging as most tiles are visually identical. ~250MB per collection, JPG

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/5cccaf94-284d-434f-a874-c071c1139c1d)

http://image-tiles.s3-website-us-east-1.amazonaws.com/bobmahar/dissections/dissection_series_c_20_20.zip
http://image-tiles.s3-website-us-east-1.amazonaws.com/bobmahar/dissections/dissection_series_d_40_40.zip 

---

**IBM Test Wafer Scan - 27,000 x 27,000 px** - Dissected into 20 x 30 tiles, overlap ~30%.
This wafer contains about 50 copies of an identical test pattern, several sevtions have very low contrast.  ~300MB per collection, JPG.

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/3d6aafe6-628f-4021-91dc-a94b8faf126e)

http://image-tiles.s3-website-us-east-1.amazonaws.com/bobmahar/dissections/dissection_series_e_20_30.zip

---

**TRW 1016 Die Photos - Original Source Images** - 76 source images, each about 130MB.   ~8.8GB for this collection.

![image](https://github.com/Bob-O-Rama/cvfind/assets/28986153/eb875563-1fb6-4960-8d16-f0c1785f1c27)

http://image-tiles.s3-website-us-east-1.amazonaws.com/bobmahar/source/TRW_1016_PMax.zip
