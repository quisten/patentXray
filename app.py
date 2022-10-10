from unittest.mock import NonCallableMock
from matplotlib.style import available
import streamlit as st
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns 
import math


dfOptions = ["Higher Than scoreTreshold", "Has Polight Trademarks", "Only PLT Competitors Related Patents", "Only PLT's' Patents"]

###################################################################################################
# Helper Functions
###################################################################################################

def dfQualifyPatents(df, scoreThres, rmPLT, rmCOMP, rmFALSE, rmNGrant, userOptions, ignoreDate, mustContain):
    st.write("Intial number of Patents: ", df.shape[0])

    # First - Remove Invalid Patents
    statusLabelDelete = ['Abandoned', 'Expired - Fee Related','Expired - Lifetime', 'Withdrawn']
    invalidStatus = df.loc[df.Status.isin(statusLabelDelete)]
    df.drop(invalidStatus.index, axis=0, inplace=True)
    st.write("After Invalid Status (Abandoned, Expired, Withdrawn) Removal: ", df.shape[0])

    # Remove Duplicates | Based on Inventor and Application Date
    df['Inventor'] = df['Inventor'].str.title()                       # Ensure good captialization
    dupTitle = df.duplicated(subset=['Inventor',  'Priority_Date'])   # Find duplicates
    dups = dupTitle.loc[dupTitle] 
    df.drop(dups.index, axis=0, inplace=True)
    st.write("After Duplicate (Kind Status) Removal: ", df.shape[0])

    # Unique Patents - Unique Title and Inventors
    #df['Inventor'] = df['Inventor'].str.title()                       # Ensure good captialization
    #dupTitle = df.duplicated(subset=['Inventor',  'Title'])   # Find duplicates
    #dups = dupTitle.loc[dupTitle] 
    #df.drop(dups.index, axis=0, inplace=True)
    #st.write("Duplicate (Title) Removal: ",df.shape[0])

    # Must have a Publication Date, Title and Inventor and "not header as row"...
    df = df.loc[df['Publication_Date'].notnull()]
    df = df[~df['Publication_Date'].str.contains('Publication_Date')]
    #df = df[~df['Publication_Date'].str.contains('/')]
    df = df.loc[df['Title'].notnull()]
    df = df.loc[df['Inventor'].notnull()]
    df['Score'] = pd.to_numeric(df['Score'])
    
    # Convert Dates
    try:
        df.Publication_Date = pd.to_datetime(df.Publication_Date, format='%Y-%m-%d')
    except:
        df.Publication_Date = pd.to_datetime(df.Publication_Date, format='%m/%d/%Y')

    try:
        df.Priority_Date = pd.to_datetime(df.Priority_Date, format='%Y-%m-%d')
    except:
        df.Priority_Date = pd.to_datetime(df.Priority_Date, format='%m/%d/%Y')

    try:
        df.Grant_Date = pd.to_datetime(df.Grant_Date, format='%Y-%m-%d')
    except:
        df.Grant_Date = pd.to_datetime(df.Grant_Date, format='%m/%d/%Y')
    
    df = df.loc[df.Publication_Date.dt.year > ignoreDate]
    df = df.loc[df['kw2'].notnull()] 
    dfAll = df.copy()
    st.write("dfAll::Has Required Data,Is newer than 2015, metions as lens: ",df.shape[0])
    

    ##########################################################
    # User Input 
    ##########################################################

    # Remove Patents from Competitors and Polight
    scoreDF = dfAll.loc[dfAll['Score'] > scoreThres]
    st.write("Has a Score Higher than ", scoreThres, ":",scoreDF.shape[0])
    
    trademarkDF = dfAll.loc[dfAll['kw1'].notnull()]
    st.write("Mentions a Polight Trademark: ",trademarkDF.shape[0])
    
    #df = dfAll.loc[(dfAll['Score'] > scoreThres) | (dfAll['kw1'].notnull())]
    #st.write("Mentions a Polight Trademark or has a higher score than", scoreThres,  ": ",df.shape[0])
    
    comp = ['Corning', 'Optotune', 'Varioptics', 'Nextlens']
    compDF = dfAll[dfAll['Assignee'].str.contains('|'.join(comp))]
    st.write("Assignee == Competition: ", compDF.shape[0])
    
    pltDF = dfAll[dfAll['Assignee'].str.contains('Polight')]
    st.write("Assignee == Polight:", pltDF.shape[0])
    
    falsePositives = ['WO2022157730A1']
    dfFalse = dfAll[dfAll['patentID'].str.contains('|'.join(falsePositives))]
    st.write("FalsePositives:", dfFalse.shape[0])
    

    # Assemble
    combineDFs = list()
    for option in userOptions:
        if option == dfOptions[0]:
            combineDFs.append(scoreDF)
        if option == dfOptions[1]:
            combineDFs.append(trademarkDF)
        if option == dfOptions[2]:
            combineDFs.append(compDF)
        if option == dfOptions[3]:
            combineDFs.append(pltDF)

    try:
        finalDF = pd.concat(combineDFs)
    except:
        finalDF = dfAll

    # Remove 
    if rmPLT: finalDF = finalDF[~finalDF.patentID.isin(pltDF.patentID)]
    if rmCOMP: finalDF = finalDF[~finalDF.patentID.isin(compDF.patentID)]
    if rmFALSE: finalDF = finalDF[~finalDF.patentID.isin(dfFalse.patentID)]
    if rmNGrant: finalDF = finalDF[finalDF['Grant_Date'].notnull()]

    #st.write(mustContain)
    #finalDF = finalDF[finalDF['Assignee'].str.contains(mustContain)]
    mustContain = mustContain.split(' ')
    mustContain = [s.capitalize() for s in mustContain]
    st.write(mustContain)
    finalDF = finalDF[finalDF['Assignee'].str.contains('|'.join(mustContain))]

    # Print some Stats
    st.write("Final DF: ", finalDF.shape[0])
  

    dfFinal = finalDF
    return (dfAll, dfFinal)

def patentTimeScatterPlotAnimation(df, filterName):


    patentTime = df.copy()
    #patentTime.Grant_Date =  pd.to_datetime(patentTime.Grant_Date, format='%Y-%m-%d')
    #patentTime.Priority_Date = pd.to_datetime(patentTime.Priority_Date, format='%Y-%m-%d')
    #patentTime.Publication_Date = pd.to_datetime(patentTime.Publication_Date, format='%Y-%m-%d')
    #patentTime = patentTime.loc[patentTime.Priority_Date.notnull()].copy() 

    patentTime["Publication_Time"] = (patentTime.Publication_Date-patentTime.Priority_Date)
    patentTime['Publication_Time'] = patentTime['Publication_Time'].dt.days
    patentTime['Region'] = patentTime['patentID'].astype(str).str[0:2]
    patentTime['Size'] = 5
    patentTime = patentTime.loc[patentTime['Region'].str.contains('|'.join(['US', 'WO', 'CN', 'KR', 'JP', 'EP']))]
    #print("%d %d" % (len(dfG1), len(patentTime)))

    fig, ax = plt.subplots(figsize = (14, 7))
    limits = pd.to_datetime(['2014-12-01 12:00','2022-06-10 12:00'])
    today = pd.to_datetime("today")
    months_ago = today-pd.DateOffset(months=22*2)
    sortedDates = patentTime.sort_values(by=['Publication_Date'], ascending=True, inplace=False).Publication_Date

    cdate = sortedDates[sortedDates.index[-1]]

    patentTime2 = patentTime.copy()
    patentTime2 = patentTime2.loc[patentTime2.Publication_Date < cdate]
    
    fig, ax = plt.subplots(figsize = (14, 7))
    plt.suptitle("Priority Date vs Publication Time [%d]" % len(patentTime2), fontsize=24)
    plt.title("FilterID: %s - Publication Dates: %s to %s " % (filterName, str(limits[0].date()), str(cdate.date())), fontsize=12)
    
    plt.grid(alpha=0.2)
    sns.scatterplot(data=patentTime2.Region, alpha=0.6, s=250, hue= patentTime.Region, x = patentTime2['Priority_Date'], y = patentTime2['Publication_Time'])
    plt.legend(loc="upper right")
    plt.xlim(left=limits[0], right=today)
    plt.ylim(bottom=0, top=1300)
    plt.plot([cdate, cdate], [0, 1300])

    #plt.plot([cdate, cdate-pd.DateOffset(months=22*2)], [0, 1300])

    # Show Time Hasn't happend yet on the vertical axis
    plt.fill_between((today, cdate), (22*30*2, 22*30*2), alpha=0.1, color='r')
    plt.fill_between((cdate,  cdate-pd.DateOffset(months=22*2)), (22*30*2, 22*30*2), (0,22*30*2), alpha=0.1, color='r')

    #plt.fill_between((today, months_ago), (22*30*2, 22*30*2), (0,22*30*2), alpha=0.1, color='r')
    #pltSaveFig(jobFolder+"Results/Animations/"+"Priority_vs_Patent_%s-%03d.png" % (fileName, cframe))
    st.pyplot(fig)
    #return

    # TODO: COPY PLOT TO Results/Plots/ and build GIF/MP4
    return



###################################################################################################
# Pages

def page_displayDataFrame(df):
    
    st.dataframe(df)

    return True

def page_keywordAnaysis(df):
    pass

def page_TopAssignees(df, filterID):
    fileName = "View Settigngs"

    with st.expander("Patents by Top Assignees", expanded=True):
        topAssignees = df.groupby(['Assignee']).size()
        topAssignees = topAssignees.loc[topAssignees.values>1]
        topAssignees = topAssignees.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize = (14, 9))
        sns.barplot(x=topAssignees.index, y=topAssignees.values)
        plt.grid(alpha=0.3)
        plt.suptitle("Patents By Assignees (more than 1)", fontsize=24)
        plt.title("Filter ID: %s" % filterID, fontsize=12)
        plt.xticks(rotation=60, ha='right')
        plt.tight_layout()
        #plt.show()
        st.pyplot(fig)
 
    with st.expander("Patent Timeline"):

        # Build Structure
        companyLabel = list()
        companyLabel2 = list()
        patentLabel = list()
        yData = list()
        yData2 = list()
        xData = list() 

        yPos = 0

        lineDatesMin = list()
        lineDatesMax = list()
        lineY = list()

        relevantPatents = list()
        for assignee in topAssignees.index: 
            patents = df.loc[df.Assignee == assignee]

            kkk = 0
            for index, patent in patents.iterrows():
                yData.append(yPos)
                xData.append(patent.Priority_Date)
                patentLabel.append(patent.patentID)
                companyLabel.append(patent.Assignee)
                yData2.append(yPos+.25*math.sin(kkk))
                kkk += 1
            lineDatesMax.append(max(patents.Priority_Date))
            lineDatesMin.append(min(patents.Priority_Date))
            lineY.append(yPos)
            companyLabel2.append(assignee[0:20])
            
            yPos += 1 

        # Plot 
        fig, ax = plt.subplots(figsize = (21, 14))
        limits = pd.to_datetime(['2015-12-01 12:00','2022-06-10 12:00'])
        today = pd.to_datetime("today")
        plt.suptitle("Timeline by Assignees", fontsize=24)
        plt.title("Filter ID: %s" % filterID, fontsize=12)
        
        plt.grid(alpha=0.2)
        sns.color_palette("Paired")
        sns.scatterplot(data=companyLabel, alpha=0.6, s=150, palette="hls", hue=companyLabel, x = xData, y = yData)

        for i in range(len(xData)):
            if xData[i].year>2012:
                plt.text(x=xData[i],y=yData2[i],s=patentLabel[i],fontdict=dict(color='black',size=7, alpha=0.25), rotation=30)

        plt.legend(loc="upper right")
        plt.xlim(left=limits[0], right=today)
        plt.ylim(bottom=-1, top=len(topAssignees.index)+1)

        lineX = list()
        linesY = list()
        lineHue = list()
        for index in range(len(lineDatesMin)):
            lineX.append(lineDatesMin[index])
            lineX.append(lineDatesMax[index])
            linesY.append(lineY[index])
            linesY.append(lineY[index])
            lineHue.append(index)
            lineHue.append(index)

        sns.lineplot(x=lineX, y=linesY, palette="hls", hue=lineHue)
        ax.set_yticks(range(0, len(topAssignees.index)))
        ax.set_yticklabels(companyLabel2)
        ax.yaxis.tick_right()
        plt.yticks(rotation=-20, ha='left')
        ax.get_legend().remove()

        st.pyplot(fig)

    return True

def page_growthAnalysis(df, filterID):

    # Prepare and Make Patent_Age
    pubStat = df.copy()
    try:
        pubStat['Priority_Date'] = pd.to_datetime(pubStat.Priority_Date, format='%Y-%m-%d')
    except:
        pubStat['Priority_Date'] = pd.to_datetime(pubStat.Priority_Date, format='%m/%d/%Y')
    
    pubStat["Patent_Age"] = (pd.to_datetime("today")-pubStat.Priority_Date)
    pubStat["Patent_Age"] = pubStat["Patent_Age"].dt.days
    pubStat = pubStat.loc[pubStat["Patent_Age"].notnull()]
    pubStat["Patent_Age"] = pubStat["Patent_Age"].astype(int)
    pubStat = pubStat.loc[pubStat["Patent_Age"] < 356*10]
    pubStat["Patent_Age_Month"] = pubStat["Patent_Age"]/30.0
    pubStat["Patent_Age_Month"] = pubStat["Patent_Age_Month"].astype(int)  

    with st.expander("Prioirty vs Publication Dates for Dataframe", expanded=True):
        patentTimeScatterPlotAnimation(pubStat, filterID)

    # Direct Trend Analysis | Score over 200 | Month 
    with st.expander("Patents Per Quarter", expanded=True):
    
        # Direct Trend Analysis | Score over 200 | Quarter """
        pubStat["Patent_Age_Quarter"] = pubStat["Patent_Age"]/90.0
        pubStat["Patent_Age_Quarter"] = pubStat["Patent_Age_Quarter"].astype(int)  
        plotData = pubStat.Patent_Age_Quarter.value_counts()
        plotData = plotData.sort_index()
        
        month, noPatents = [], []
        for m in range(pubStat["Patent_Age_Quarter"].max()):
            month.append(m)
            noPatents.append(len( pubStat.loc[pubStat["Patent_Age_Quarter"]==m] ))

        fig, ax = plt.subplots(figsize = (14, 7))
        sns.regplot(x=month, y=noPatents)
        plt.xticks(rotation=60, ha='right')
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left")
        plt.xlabel("Patent Age (in Quarters)", fontsize=12)
        plt.ylabel("Number of Patents", fontsize=12)
        plt.suptitle("Patents that Satisfies Filter Requirements [%d]" % len(df), fontsize=24)
        plt.title("filterID: %s" % filterID)
        plt.fill_between((0, 18/3), 0, 10, alpha=0.1, color='r')
        st.pyplot(fig)
        #pltSaveFig(outputFolder+"TrendAnalysis_Quarterly_%s.png" % fileName)

    with st.expander(label="Patent Time (From filed to public) as a Distribution ", expanded=False):
        #Returns an lookuptable where index is month and value is estimated patent saturation"""
        #pubStat = df.copy()

        try:
            pubStat.Priority_Date = pd.to_datetime(pubStat.Priority_Date, format='%Y-%m-%d')
        except:
            pubStat.Priority_Date = pd.to_datetime(pubStat.Priority_Date, format='%m/%d/%Y')
        try:
            pubStat.Publication_Date = pd.to_datetime(pubStat.Publication_Date, format='%Y-%m-%d')
        except: 
            pubStat.Publication_Date = pd.to_datetime(pubStat.Publication_Date, format='%m/%d/%Y')
        
        pubStat["Publication_Time"] = (pubStat.Publication_Date-pubStat.Priority_Date)
        pubStat['Publication_Time'] = pubStat['Publication_Time'].dt.days
        pubStat['Publication_Time'] = pd.cut(pubStat['Publication_Time'], bins=[i*30 for i in range(33)])
        plotData = pubStat.Publication_Time.value_counts()/len(pubStat)
        
        fig, ax = plt.subplots(figsize = (14, 7))
        fig.subplots_adjust(bottom=0.2)
        sns.barplot(x=plotData.index, y=plotData.values)
        plt.grid(alpha=0.3)
        plt.title("PatentTime Distribution by Months Time [%d]" % len(pubStat), fontsize=24)
        plt.xticks(rotation=60, ha='right')
        st.pyplot(fig)
        #pltSaveFig(outputFolder+"patentTime_Lookup_Normalized.png")

        additive = 0.0
        index, cumSum = [], []
        plotData = plotData.sort_index()
        for i,v in enumerate(plotData.values):
            plotData.values[i] = i
            index.append(i)
            additive += v
            cumSum.append(additive)
                
        fig, ax = plt.subplots(figsize = (14, 7))
        fig.subplots_adjust(bottom=0.2)
        sns.barplot(x=index, y=cumSum)
    
        plt.grid(alpha=0.3)
        plt.title("PatentTime Distribution by Months Time [%d]" % len(pubStat), fontsize=24)
        plt.xticks(rotation=60, ha='right')
        st.pyplot(fig)
        #pltSaveFig(outputFolder+"patentTime_Lookup_Normalized_CumSum.png")

    with st.expander(label="Optimistic Upside Prediction", expanded=True):
        patentProb = cumSum
        #""" Adjusted Regplot | Month """
        #patentProb = patentDistributionCurve(df, outputFolder)

        index, actual, estimate = [], [], []
        for i in range(len(patentProb)):
            xx = len(pubStat.loc[pubStat['Patent_Age_Month'] == i])
            if xx==0:
                continue
            index.append(i)
            actual.append(xx)
            estimate.append(actual[-1]/patentProb[i]) 

        plotData = pubStat.Patent_Age_Month.value_counts()
        plotData = plotData.sort_index()
        fig, ax = plt.subplots(figsize = (14, 7))
        #sns.regplot(x=index, y=actual, label="actual")
        sns.regplot(x=index, y=estimate, label="estimate")
        sns.regplot(x=plotData.index, y=plotData.values, label="actual") 
        plt.xticks(rotation=60, ha='right')
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left")
        plt.xlabel("Patent Age (in months)", fontsize=12)
        plt.ylabel("Number of Patents", fontsize=12)
        plt.suptitle("Patents That Satisfies Filter Requirements [%d]" % len(df), fontsize=24)
        plt.title("filterID: %s" % filterID)
        plt.fill_between((0, 18), 0, 10, alpha=0.1, color='r')
        plt.ylim(0, 13)
        st.pyplot(fig)
        #pltSaveFig(outputFolder+"PredictedGrowth_Montly_%s.png" % fileName)


        #""" Cumulative of both - going backwards """
        
        cest, cact = [0], [0]
        cindex = []
        for j in range(len(estimate)):
            cest.append(cest[-1]+estimate[-1-j])
            cact.append(cact[-1]+actual[-1-j])
            cindex.append(index[-1-j])


        fig, ax = plt.subplots(figsize = (14, 7))
        sns.lineplot(x=list(range(len(cest), 0, -1)), y=cest, label="estimate")
        sns.lineplot(x=list(range(len(cact), 0,-1)), y=cact, label="actual") 
        plt.xticks(rotation=60, ha='right')
        plt.grid(alpha=0.3)
        
        #plt.ylim((0, 12))
        plt.legend(loc="upper right")
        plt.xlabel("Patent Age (in months)", fontsize=12)
        plt.ylabel("Number of Patents", fontsize=12)
        plt.suptitle("Cumulative Growth and Probabilities [%d]" % len(df), fontsize=24)
        plt.title("filterID: %s" % filterID)
        
        st.pyplot(fig)
        #pltSaveFig(outputFolder+"PredictedGrowth_Cum_Adjusted__zeros_Direct_Month_TrendAnalysis_G3_200_TunableOptics.png")


    return True

###################################################################################################
def main():

    # Primary Layout 
    st.title("Patent Analysis")
    st.write("Analysis of Patents related to Tunable Optics Searches")
   
    availableRuns = os.listdir("./Runs")

    with st.expander("Select Data To Process", expanded=True):
        selectedRun = st.selectbox(label="Select Run", options=availableRuns)
        #rmPLT = st.checkbox(label="Remove Polight's Patents", value=True)
        
        scoreTresh = st.slider("Score Treshold", min_value=0, max_value=600, value=100)
  
        ## Create DF 
        df = pd.read_csv('./Runs/%s/' % selectedRun+"crawlerALL_translated.csv", index_col=[0], encoding='utf-8')

    with st.expander("Advanced Settings", expanded=True):
        
        ignoreDate = st.slider("Remove Patents Older Than", min_value=2000, max_value=pd.to_datetime("today").year, value=2015)
        mustContain = st.text_input("'Assignees' Must Contain:")

        dfPreset = st.multiselect("Dataframe Preset", dfOptions, default=dfOptions[0])
        rmPLT = st.checkbox(label="Remove Polight's Patents", value=True)
        rmCOMP = st.checkbox(label="Remove Competitor's Patents", value=True)
        rmFALSE = st.checkbox(label="Remove Known False-Positives", value=True)
        rmNGrant = st.checkbox(label="Remove Not Granted", value=False)
        st.write("Competitors: Varioptics, Corning, Optotune, Nextlens")

    with st.expander("Debug Information of Dataframe Generation", expanded=False):
        (dfAll, dfFinal) = dfQualifyPatents(df, scoreTresh, rmPLT, rmCOMP, rmFALSE, rmNGrant, dfPreset, ignoreDate, mustContain)    

        # Create Search-Identifier
        presetIDs = '|'.join([str(dfOptions.index(s)) for s in dfPreset])
        exlude = ['X', 'X', 'X', 'X']
        for i, e in enumerate([rmPLT, rmCOMP, rmFALSE, rmNGrant]):
            if e:
                exlude[i] = 'O' 
        exlude = '|'.join(exlude)
        filterID = selectedRun+'+'+str(scoreTresh)+'+'+presetIDs+'+'+exlude+'+'+mustContain

        st.write("Filter ID:", filterID)
    
    st.write("DataFrame Size: %s" % len(dfFinal))

    tab1, tab2, tab3, tab4 = st.tabs(["DataFrame", "Top Assignees", "Patent Growth", "Keyword Analysis", ])

    with tab1:
        page_displayDataFrame(dfFinal)
    with tab2:
        page_TopAssignees(dfFinal, filterID)
    with tab3:
        page_growthAnalysis(dfFinal, filterID)
    with tab4:
        pass


    st.write("Version 1.0 alpha - 10 October 2022")
    
    return True

###################################################################################################
# Init 
if __name__ == "__main__":
    st.set_page_config(
        "Patent Analyiser",
        "📊",
        initial_sidebar_state="expanded",
        #layout="wide",
    )
    main()
