import os
import pandas as pd
import matplotlib.pyplot as plt

data_csv_path = f"..{os.sep}Data{os.sep}prepared_dataframe.csv" 
df = pd.read_csv(data_csv_path)

if not os.path.exists(fr"DataVisPlots"):
    os.makedirs(fr"DataVisPlots")

# Emotion - Gender
plt.clf()
df1=df.groupby(['emotion','gender']).size()
df1=df1.unstack()
df1.plot(kind='bar', figsize=(12,10))
plt.savefig(fr"DataVisPlots{os.sep}EmotionGender.png",dpi=300)

# Emotion - Statement
plt.clf()
df2=df.groupby(['emotion','statement']).size()
df2=df2.unstack()
df2.plot(kind='bar', figsize=(12,10))
plt.savefig(fr"DataVisPlots{os.sep}EmotionStatement.png",dpi=300)

# Emotion - Emotional Intensity
plt.clf()
df3=df.groupby(['emotion','emotional_intensity']).size()
df3=df3.unstack()
df3.plot(kind='bar', figsize=(12,10))
plt.savefig(fr"DataVisPlots{os.sep}EmotionEmIntensity.png",dpi=300)

# Mean Centroid - Female/Male
plt.clf()
plt.plot(df["mean_centroid"][df["gender"]=="female"], "bx", label="Female")
plt.plot(df["mean_centroid"][df["gender"]=="male"], "r+", label="Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderMeanCentroid.png",dpi=300)

# STD Centroid - Female/Male
plt.clf()
plt.plot(df["std_centroid"][df["gender"]=="female"], "bx", label="Female")
plt.plot(df["std_centroid"][df["gender"]=="male"], "r+", label="Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderStdCentroid.png",dpi=300)

# Mean Bandwidth - Female/Male
plt.clf()
plt.plot(df["mean_bandwidth"][df["gender"]=="female"], "bx", label="Female")
plt.plot(df["mean_centroid"][df["gender"]=="male"], "r+", label="Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderMeanBandwidth.png",dpi=300)

# STD Bandwidth - Female/Male
plt.clf()
plt.plot(df["std_bandwidth"][df["gender"]=="female"], "bx", label="Female")
plt.plot(df["std_bandwidth"][df["gender"]=="male"], "r+", label="Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderStdBandwidth.png",dpi=300)

# Mean Centroid - Calm/Angry
plt.clf()
plt.plot(df["mean_centroid"][df["emotion"]=="calm"], "bx", label="Calm")
plt.plot(df["mean_centroid"][df["emotion"]=="angry"], "r+", label="Angry")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}CalmAngryMeanCentroid.png",dpi=300)

# STD Centroid - Calm/Angry
plt.clf()
plt.plot(df["std_centroid"][df["emotion"]=="calm"], "bx", label="Calm")
plt.plot(df["std_centroid"][df["emotion"]=="angry"], "r+", label="Angry")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}CalmAngryStdCentroid.png",dpi=300)

# Mean Bandwidth - Emotions
plt.clf()
plt.plot(df["mean_bandwidth"][df["emotion"]=="calm"], "x", label="calm")
plt.plot(df["mean_bandwidth"][df["emotion"]=="angry"], "x", label="angry")
plt.plot(df["mean_bandwidth"][df["emotion"]=="fearful"], ".", label="fearful")
plt.plot(df["mean_bandwidth"][df["emotion"]=="disgust"], ".", label="disgust")
plt.plot(df["mean_bandwidth"][df["emotion"]=="surprised"], ".", label="surprised")
plt.plot(df["mean_bandwidth"][df["emotion"]=="neutral"], "*", label="neutral")
plt.plot(df["mean_bandwidth"][df["emotion"]=="happy"], "^", label="happy")
plt.plot(df["mean_bandwidth"][df["emotion"]=="sad"], "^", label="sad")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}EmotionMeanBandwidth.png",dpi=300)

# Mean Centroid - Emotions
plt.clf()
plt.plot(df["mean_centroid"][df["emotion"]=="calm"], "x", label="calm")
plt.plot(df["mean_centroid"][df["emotion"]=="angry"], "x", label="angry")
plt.plot(df["mean_centroid"][df["emotion"]=="fearful"], ".", label="fearful")
plt.plot(df["mean_centroid"][df["emotion"]=="disgust"], ".", label="disgust")
plt.plot(df["mean_centroid"][df["emotion"]=="surprised"], ".", label="surprised")
plt.plot(df["mean_centroid"][df["emotion"]=="neutral"], "*", label="neutral")
plt.plot(df["mean_centroid"][df["emotion"]=="happy"], "^", label="happy")
plt.plot(df["mean_centroid"][df["emotion"]=="sad"], "^", label="sad")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}EmotionMeanCentroid.png",dpi=300)

# Mean Centroid - Female/Male - Angry
plt.clf()
plt.plot(df["mean_centroid"][(df["gender"]=="female") & (df["emotion"]=="angry")], "bx", label="Angry Female")
plt.plot(df["mean_centroid"][(df["gender"]=="male") & (df["emotion"]=="angry")], "r+", label="Angry Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderAngryMeanCentroid.png",dpi=300)

# Mean Centroid - Female/Male - Calm
plt.clf()
plt.plot(df["mean_centroid"][ (df["gender"]=="female") & (df["emotion"]=="calm")], "bx", label="Calm Female")
plt.plot(df["mean_centroid"][ (df["gender"]=="male") & (df["emotion"]=="calm")], "r+", label="Calm Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderCalmMeanCentroid.png",dpi=300)

# Mean Centroid/Bandwidth - Female/Male - Angry
plt.clf()
plt.plot(df["mean_centroid"][ (df["gender"]=="female") & (df["emotion"]=="angry")], df["mean_bandwidth"][ (df["gender"]=="female") & (df["emotion"]=="angry")], "bx", label="Angry Female")
plt.plot(df["mean_centroid"][ (df["gender"]=="male") & (df["emotion"]=="angry")],  df["mean_bandwidth"][ (df["gender"]=="male") & (df["emotion"]=="angry")], "r+", label="Angry Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderAngryMeanCentroidBand.png",dpi=300)

# Mean Centroid/Bandwidth - Female/Male - Angry
plt.clf()
plt.plot(df["mean_centroid"][ (df["gender"]=="female") & (df["emotion"]=="calm")], df["mean_bandwidth"][ (df["gender"]=="female") & (df["emotion"]=="calm")], "bx", label="Calm Female")
plt.plot(df["mean_centroid"][ (df["gender"]=="male") & (df["emotion"]=="calm")],  df["mean_bandwidth"][ (df["gender"]=="male") & (df["emotion"]=="calm")], "r+", label="Calm Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderCalmMeanCentroidBand.png",dpi=300)


# Mean Centroid/Bandwidth - Female/Male
plt.clf()
plt.plot(df["mean_centroid"][df["gender"]=="female"], df["mean_bandwidth"][df["gender"]=="female"], "bx", label="Female")
plt.plot(df["mean_centroid"][df["gender"]=="male"], df["mean_bandwidth"][df["gender"]=="male"], "r+", label="Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderMeanCentroidBand.png",dpi=300)

# STD Centroid/Bandwidth - Female/Male
plt.clf()
plt.plot(df["std_centroid"][df["gender"]=="female"], df["std_bandwidth"][df["gender"]=="female"], "bx", label="Female")
plt.plot(df["std_centroid"][df["gender"]=="male"],  df["std_bandwidth"][df["gender"]=="male"], "r+", label="Male")
plt.legend()
plt.savefig(fr"DataVisPlots{os.sep}GenderStdCentroidBand.png",dpi=300)