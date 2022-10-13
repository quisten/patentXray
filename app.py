#from unittest.mock import NonCallableMock
#from matplotlib.style import available
import streamlit as st
import pandas as pd
import os

import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns 
import math
import io

hvar = """  <script>
                    var elements = window.parent.document.querySelectorAll('.streamlit-expanderHeader');
                    var arrayLength = elements.length;
                    for (var i =0 ; i< arrayLength; i++) {
                        elements[i].style.color = 'rgba(183, 56, 68, 1)';
                        elements[i].style.fontSize = 'normal';
                        elements[i].style.fontWeight = 'normal'; 
                    }
            </script>"""

dfOptions = ["Higher Than ScoreTreshold", "Contains poLight Trademarks", "Patents by Varioptics/Cornings", "Patents by poLight"]

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
    #combineDFs.append(scoreDF)

    st.write("dfFinal=")
    
    for option in userOptions:
        if option == dfOptions[0]:
            combineDFs.append(scoreDF)
            st.write("+", scoreDF.shape[0])
        if option == dfOptions[1]:
            combineDFs.append(trademarkDF)
            st.write("+", trademarkDF.shape[0])
        if option == dfOptions[2]:
            combineDFs.append(compDF)
            st.write("+", compDF.shape[0])
        if option == dfOptions[3]:
            combineDFs.append(pltDF)
            st.write("+", pltDF.shape[0])

    try:
        finalDF = pd.concat(combineDFs).drop_duplicates().reset_index(drop=True)
    except:
        finalDF = dfAll

    st.write("finalDF before Removal:", finalDF.shape[0])
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

def getCumulativeDateTimeLine(subsetDates): 
    plotSerie = subsetDates.copy()
    
    try:
        plotSerie.index = pd.to_datetime(plotSerie.values, format="%Y-%m-%d")
    except:
        plotSerie.index = pd.to_datetime(plotSerie.values, format="%m/%d/%Y")
    
    plotSerie = plotSerie.sort_index(ascending=True)
    
    #plotSerie.values[:] = str("A")#100#"A"# int(1)
    plotSerie = pd.Series(data=[1 for x in range(len(plotSerie))], index=plotSerie.index)
    plotSerie.values[:] = plotSerie.values.cumsum()
    #print(plotSerie)
    #st.write(plotSerie)
    return plotSerie

###################################################################################################
# Plots
###################################################################################################

###################################################################################################
# Pages
###################################################################################################

def page_displayDataFrame(df, filterID):

    # Modify URL
    df_mod = df.copy()
    
    #df_mod['URL'] = df_mod['URL'].str.split('"')
    #df_mod['URL'] = df_mod['URL'].str.replace("\"=HYPERLINK\(", "")
    #df_mod['URL'] = df_mod['URL'].str.replace("\"\)", "")

    # Display 
    st.dataframe(df_mod)

    @st.cache
    def convert_df(df): 
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    fileName = filterID.replace('+','_').replace('|', '')+'.csv'
    st.download_button("Download Dataframe as CSV", csv, fileName, "text/csv", key='download-csv')

    return True

def page_keywordAnaysis(df, filterID):
    
    if True:
    #with st.expander(label="Most often Used Lense Descriptors"):    
        
        relatedKeywords = ['tunable', 'adjustable', 'variable', 'deformable']

        fig, ax = plt.subplots(figsize = (20, 7))
        for i, kwd in enumerate(relatedKeywords):
            plotData = df.copy()
            plotData = plotData.loc[plotData['kw2'].notnull()]
            plotData = plotData.loc[plotData['kw2'].str.contains(kwd)]

            tunable = getCumulativeDateTimeLine(plotData.Priority_Date)             # Change this! 
            plt.plot(tunable.index, tunable.values, alpha=.5, label=kwd)

        if all:
            plotData = df.copy()
            plotData = plotData.loc[plotData['kw2'].notnull()]
            plotData = plotData.loc[plotData['kw2'].str.contains('|'.join(relatedKeywords))]
            tunable = getCumulativeDateTimeLine(plotData.Priority_Date)             # Change this! 
            plt.plot(tunable.index, tunable.values, label='All keywords', linestyle='dashed', color=sns.color_palette("tab10")[4])
            #plotData.to_csv(outputFolder+"tunableTrend_published.csv", encoding='utf-8')
        
        limits = pd.to_datetime(['2015-01-10 12:00','2022-01-10 12:00'])
        plt.xlim(limits[0], limits[1])
        plt.xticks(rotation=90, ha='right')
        plt.grid(alpha=0.2)
        plt.legend(loc="upper left")
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Number of Patents", fontsize=12)
        plt.suptitle("Patents describing new Lens Technology", fontsize=24)
        plt.title("FilterID: %s" % filterID)
        st.pyplot(fig)

    if True:  
    #with st.expander(label="Popular Lens Descriptors"):

        showUnique = st.checkbox(label="Count Adjective Once Per Patent (Unique):", value=True)

        uniqueKeywords = {}
        totalKeywords = {}
        for index, row in plotData.iterrows():
            #print(row.kw2)
            for tech in row.kw2.split(','):
                inner = tech.split('|')
                try:
                    uniqueKeywords[inner[0]] += 1
                except:
                    uniqueKeywords[inner[0]] = 1

                try:
                    totalKeywords[inner[0]] += int(inner[1])
                except:
                    totalKeywords[inner[0]] = int(inner[1])

        #print(uniqueKeywords)
        #print(totalKeywords)
        #st.write(uniqueKeywords)
        
        fig, ax = plt.subplots(figsize = (20, 7))

        #plt.bar(x=[i for i,x in enumerate(uniqueKeywords)], height=uniqueKeywords.values())
        if showUnique:
            plt.bar(x=tuple(uniqueKeywords.keys()), height=uniqueKeywords.values())  
            plt.xticks(rotation=60, ha='right')
            plt.suptitle("Unique Keywords used Dataset [%d of %d]" % (len(tunable), len(df)))
            plt.title("FilterID: %s" % filterID)
            plt.xlabel("", fontsize=12)
            plt.ylabel("Unique Mentions (More than one kws per patent)", fontsize=12)
            plt.tight_layout()
        else:
            plt.bar(x=totalKeywords.keys(), height=totalKeywords.values())
            plt.xticks(rotation=60, ha='right')
            plt.suptitle("Unique Keywords Dataset [%d of %d]" % (len(tunable), len(df)))
            plt.title("FilterID: %s" % filterID)
            plt.ylabel("Unique Mentions (More than one kws per patent)", fontsize=12)
            plt.tight_layout()
        
        st.pyplot(fig)

    return True

def page_TopAssignees(df, filterID):
    fileName = "View Settigngs"


    #with st.beta_expander("Patents by Top Assignees", expanded=True):
    if True: 
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
        st.pyplot(fig)
        st.write("Display all Assignees in the dataframe that has more than 1 patent that satisfies the current selection rules")
        
        # New way
        #img = io.BytesIO()
        #plt.savefig(img, format='png')
        #st.image(img, caption="Display assignees that has more than 1 patent in the current filter")
        #st.image("https://placekitten.com/500/300")


    st.markdown("""---""")
    #with st.expander("Patent Timeline"):
    if True:
        
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
        st.write("Show when assignees with more than 1 patent _filed_ (Priority Date) for their patents. Remember that the typical patent time in US/KR/WO is much longer than in China.")

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

    if True:
    #with st.expander("Prioirty vs Publication Dates for Dataframe", expanded=True):
        patentTimeScatterPlotAnimation(pubStat, filterID)
        st.write("By Using Publication Date on the X-axis and days from Filing to Publication (PatentTime = Publication_Date-Priority_Date) we get an insight how the activities in the different regions develop.") 

    st.markdown("""---""")          
    # Direct Trend Analysis | Score over 200 | Month 
    if True:
    #with st.expander("Patents Per Quarter", expanded=True):
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
        st.write("Patents Per Quarter - and drawing a regression line through it.")

        #pltSaveFig(outputFolder+"TrendAnalysis_Quarterly_%s.png" % fileName)
    st.markdown("""---""")
    
    if True:
    #with st.expander(label="Patent Time (From filed to public) as a Distribution ", expanded=False):
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
        st.write("Looking at Patent Time Distributions. They can be extraploated to give an non-conservative indication of what migth come.")

        #pltSaveFig(outputFolder+"patentTime_Lookup_Normalized_CumSum.png")
    st.markdown("""---""")
    if True:
    #with st.expander(label="Optimistic Upside Prediction", expanded=True):
     
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
        st.write("This is the distributions applied. Highly uncertain - but a fun plots to show! It's a straight up multiplication of 'how many patents today' by 'how many \% of patens have we seen based on todays date'. Super inaccurate and numerically unstable.")

    return True

def page_keywordXRay(df, filterID):
    
    #with st.expander("Keyword XRay", expanded=True):
    if True:     
        st.write("Show occurences of said keyword group as a function of assignees. Not all patents uses keywords as active, some even tries to fully avoid it.")

        options = ['Kw1: Trademarks', 'Kw2: Lens Adjectives', 'Kw3: Components', 'Kw4: Competitive Solutions', 'Kw5: Use-Cases']
        option = st.selectbox("Select Keywords", options, index=4)
        rowToUse = 'kw'+str(options.index(option)+1)
        #st.write(rowToUse)

        byAssignee = {}
        plotData = df.copy()
        plotData = plotData.loc[plotData[rowToUse].notnull()]
        #print("%s length = %d" % (graphTitle, len(plotData)))
        for index, row in plotData.iterrows():
            byAssignee[row.Assignee] = {}
        for index, row in plotData.iterrows():
            for app in row[rowToUse].split(','):
                inner = app.split('|')
                
                if inner[0] == 'augmented':
                    inner[0] = 'AR'

                #try:
                if inner[0] in byAssignee[row.Assignee].keys():
                    byAssignee[row.Assignee][inner[0]] += 1
                else:
                    byAssignee[row.Assignee][inner[0]] = 1
            
            # except:
            #     byAssignee[row.Assignee] = {} 
            #     byAssignee[row.Assignee][inner[0]] = 1
                    
        #print(byAssignee)        
        # byAssignee
        fig, ax = plt.subplots(figsize = (20, 20))
        fig.subplots_adjust(bottom=0.5)
        plt.grid(alpha=0.2)
        plt.suptitle("%s by Assignee" % option, fontsize=18)
        plt.title("filterID: %s" % filterID, fontsize=18)
        df2 = pd.DataFrame(data=byAssignee)
        sns.heatmap(data=df2, annot=True, linecolor="black")
        plt.xticks(rotation=60, ha='right')
        st.pyplot(fig)

    return True

###################################################################################################
def main():


    # Always show fullscreen
    st.markdown(
        """
    <style>
    button {
        opacity: 1 !important;
        transform: scale(1) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Primary Layout 
    st.title("Patent XRay")
    st.write("Analysis of Patents related to the future of Tunable Optics.")    
    st.write("Find the Instuctions and Disclaimer on the bottom.")
   
    availableRuns = os.listdir("./Runs")
    availableRuns.sort()

    with st.expander("Select Data To Process", expanded=True):
        selectedRun = st.selectbox(label="Select Run", options=availableRuns, index=len(availableRuns)-1)
        #rmPLT = st.checkbox(label="Remove Polight's Patents", value=True)
        
        scoreTresh = st.slider("Score Treshold", min_value=0, max_value=600, step=5, value=200)
  
        ## Create DF 
        df = pd.read_csv('./Runs/%s/' % selectedRun+"crawlerALL_translated.csv", index_col=[0], encoding='utf-8')

    with st.expander("Advanced Settings", expanded=False):
        
        ignoreDate = st.slider("Remove Patents Older Than", min_value=2000, max_value=pd.to_datetime("today").year, value=2015)
        mustContain = st.text_input("'Assignees' Must Contain:")

        dfPreset = st.multiselect("Dataframe Subsets (to Combine)", dfOptions, default=dfOptions[0:2])
        rmPLT = st.checkbox(label="Remove poLight's Patents", value=True)
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
    
    st.write("Number of Patents: %s" % len(dfFinal))
    
    generateCharts = False
    if st.button("Generate Charts"):
        generateCharts = True
  

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["DataFrame", "Top Assignees", "Patent Growth", "Lens Adjective Analysis", "Keyword X-Ray"])

    
    with tab1:
        page_displayDataFrame(dfFinal, filterID)

        mustContainID = st.text_input("Find PatentCode:")
        selection = df[df['patentID'].str.contains(mustContainID)]
        if mustContainID != "":
            st.dataframe(selection)
    
    with tab2:
        if generateCharts:
            page_TopAssignees(dfFinal, filterID)
        else:
            st.write("You Must Generate Charts")
    with tab3:
        if generateCharts:
            page_growthAnalysis(dfFinal, filterID)
        else:
            st.write("You Must Generate Charts")
    with tab4:
        if generateCharts:
            page_keywordAnaysis(dfFinal, filterID)
        else:
            st.write("You Must Genearte Charts")
    with tab5:
        #if generateCharts:
        page_keywordXRay(dfFinal, filterID)
        #else:
        #    st.write("You Must Genearte Charts")

    st.markdown("""---""")

    with st.expander("Instructions", expanded=False):
        st.markdown("**About**")
        markdown = """This application is made for analysing collected patent-databased that has been created by scraping the forums on SV/FA/Reddit.
                      Patents has been subjected to a keyword analysis that includes:</p>
                      - 1. polight trademarks 
                      - 2. lense adjectives 
                      - 3. technical elements/descriptions 
                      - 4. competitive tunable components, 
                      - 5. use-cases.
                      A score is given based on the keyword analysis. The database is built by traversing the citations of intereting/relevant patents.
                      Patent XRay allows you to select and combine subsets from these seaches known as _'Runs'_ by selecting _'scoreTreshold'_.
                      More advacend settings is avaiable which allows you to scrutinize competitive patetns, poLights patents and only the patents that includes trademarks.
                      More advacend settings is avaiable which allows you to scrutinize competitive patetns, poLights patents and only the patents that includes trademarks."""
        st.markdown(markdown)

        st.markdown("**Score**")
        st.markdown("The score is an arbitarily value that represent the presence of keywords related to Tunable Optics and poLight trademarks. A score of ~100 suggests a high probability that the patent includes a tunable optic solution.")
        st.markdown("A score of ~200 indicates a very high probability that specifically TLens is described or directly mentioned in the patent.")

        st.markdown("**Use-cases**")
        st.markdown("1. Look-up specific patents: Under the Dataframe tab you can lookup a specific patentCode to see if it exists in the database. If yes, then you get to see the score and all the keywords accociated with said patent.")
        st.markdown("2. Find most relevant patents: Use default values to find the most interesting patents to read and companies to investigate.")
        st.markdown("3. Use 'Assignee' under advanced settings to isolate patents from f.ex Sunny and Truly by using 'Sunny Truly' patents. NB: All names are capitalized!")
        st.markdown("All selection can be downloaded as CSV files")

        st.markdown("**Tips**")
        st.markdown("Mobile Version - The DataFrame viewer is hard  to use on android but really great on Desktop.")
        st.markdown("Mobile Version - Swipe left to get Fullscreen-View for graphs")

        st.markdown("**Want to help?**")
        st.markdown("Information about false-positives and other patents you think have been rated falsely is crucial for the improvement of these tools.")
        st.markdown("I am especially in the need for some CSS skills to make this look pretty! Hit me up if you're up for the job!")

        st.video(data="Data/overview.mp4", format="video/mp4")

        #st.write("This Application allows analysis on a patent databases. Databases are created by collecting patents shared on various forums, SV/FA/Reddit. These are given a score based on occurrences of keywords, if the score is high enough, citations will also be evaluted. The mention of polymer, membranes and autofocus will usually yield in high values.")
        #st.write("Basic Usage: _Select a 'scoreThreshold' in 'Select Data...'_")
        #st.write("Advanced Usage: _Select one or more subsets and remove patents by the use of special attributes based on date, assignees and granted status._")
        #st.write("_This App renderes best on desktop, especially the DataFrame viewer which is actually kinda great._")
        #st.write("_On Mobile you can swipe left to get full-size view option for all plots._")


    
    st.write("**Disclaimer**: This application is purely made for entertainment and educational purposes. Use it identify and scrutinize patents, companies and technology. Errors in underlying data, method and programming is guaranteed, the author is unable to find them.")
    st.write("_'The world is your oyster but you're allergic to shellfish.'_")
    st.markdown("""---""")
    st.write("Version 1.2 alpha - 12 October 2022")
    components.html(hvar, height=0, width=0)

    return True

###################################################################################################
# Init 
if __name__ == "__main__":
    st.set_page_config(
        "Patent Crawler XRay",
        "📊",
        initial_sidebar_state="expanded",
        #layout="wide",
    )
    main()
