# 评测报告 · `llm_only.ollama.qwen3_8b`

## 一、总览

### 推理阶段

| 指标 | 数值 |
|---|---|
| 样本数 | 500 |
| 总耗时 | 3 时 36 分 4 秒 |
| 吞吐 | 0.04 题/秒 |
| 输入 Token | 41,778 |
| 输出 Token | 427,800 |
| 合计 Token | 469,578 |
| 平均每题 Token | 939.2 |

### 综合指标

| 指标 | 数值 |
|---|---|
| 精确匹配 EM | 0.0500 |
| F1 | 0.1342 |
| ROUGE-L | 0.1329 |
| 平均延迟 (秒) | 25.93 |

## 二、耗时分布（秒）

| 统计量 | 数值 |
|---|---|
| 平均 | 25.93 |
| 最小 | 2.42 |
| 最大 | 80.79 |
| P50  | 21.22 |
| P95  | 66.68 |
| 合计 | 12963.0 |

## 三、Token 分布

| 字段 | 合计 | 平均 | 最小 | 最大 | P50 | P95 |
|---|---|---|---|---|---|---|
| 输入 Token | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 输出 Token | 0 | 0.0 | 0 | 0 | 0 | 0 |
| 合计 Token | 0 | 0.0 | 0 | 0 | 0 | 0 |

## 五、最差样本 Top-10（按 F1）

| 编号 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) |
|---|---|---|---|---|---|---|
| 3hop1__147162_131905_41948 | Besides the body of water by the city where Erwin Speckler d… | English Channel | sea level | 0.000 | 0.000 | 52.271 |
| 3hop1__379315_160713_77246 | What is the Arabic dictionary meaning of the word referring … | the country of India | Hindu | 0.000 | 0.000 | 13.152 |
| 3hop2__57238_1926_54362 | Who burned down the city where Keeping Up with the Joneses i… | Confederate Gen. John Bell Hood | Richmond | 0.000 | 0.000 | 57.214 |
| 2hop__853227_68633 | Arthur Marryatt is a member of a sports association that has… | Thomas Bach | Sir Clive Woodward | 0.000 | 0.000 | 12.697 |
| 3hop1__19768_19788_15107 | What percentage of Eritrea is estimated to adhere to the rel… | 48% | 50% | 0.000 | 0.000 | 49.105 |
| 3hop1__632623_160713_77246 | In the arabic dictionary what is the meaning of the word tha… | the country of India | Hindu | 0.000 | 0.000 | 57.463 |
| 3hop2__792915_467331_162182 | In what region of the country containing the Cái River is th… | South Central Coast | Red River Delta | 0.000 | 0.000 | 7.558 |
| 2hop__159215_779396 | Where was the person who wrote about the rioting being a div… | University of Glasgow | Morehouse College | 0.000 | 0.000 | 6.756 |
| 3hop1__389955_132457_47686 | Where do Greyhound buses leave from in the city where the ba… | Toronto Coach Terminal | Salt Lake City International Airport | 0.000 | 0.000 | 24.972 |
| 2hop__158275_35739 | When was one internet browser's version of Windows 8 made ac… | September 2013 | 2012 | 0.000 | 0.000 | 15.167 |

## 六、最好样本 Top-10（按 F1）

| 编号 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) |
|---|---|---|---|---|---|---|
| 2hop__622637_120171 | What year did the company that published Starship Command en… | 1986 | 1986 | 1.000 | 1.000 | 6.482 |
| 2hop__821197_368148 | In which county is Southern Maryland Electric Cooperative he… | Charles County | Charles County | 1.000 | 1.000 | 5.239 |
| 2hop__90773_715759 | What was the inventor of the atomic bomb trigger a participa… | Manhattan Project | Manhattan Project | 1.000 | 1.000 | 2.788 |
| 2hop__42578_55840 | Who played the monarch at the end of WW2 in The King's Speec… | Colin Firth | Colin Firth | 1.000 | 1.000 | 6.323 |
| 2hop__458672_20057 | Who started the foundation that developed the Free Software … | Richard Stallman | Richard Stallman | 1.000 | 1.000 | 4.579 |
| 3hop1__160324_131905_41948 | What body of water was dry land during the last ice age, bes… | English Channel | English Channel | 1.000 | 1.000 | 22.782 |
| 2hop__30905_213732 | Who is the child of the person who sponsored the resolution … | Meghan McCain | Meghan McCain | 1.000 | 1.000 | 13.962 |
| 2hop__28203_55840 | Who played king whose regal Name did Albert Adopt in the kin… | Colin Firth | Colin Firth | 1.000 | 1.000 | 5.320 |
| 2hop__52283_75184 | Who play's Marty's girlfriend in back to the future? | Claudia Wells | Claudia Wells | 1.000 | 1.000 | 9.879 |
| 3hop1__829944_474097_84283 | Who is the owner of the record label that the performer of B… | Warner Music Group | Warner Music Group | 1.000 | 1.000 | 30.956 |

## 七、全部 500 条样本

<details><summary>点击展开完整列表</summary>

| 编号 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) |
|---|---|---|---|---|---|---|
| 2hop__85713_72187 | The actor who played Thelma in "Thelma and Louise" also played which character in "A League of Their Own"? | Dottie Hinson | Dottie Hinson | 1.000 | 1.000 | 8.348 |
| 3hop1__147162_131905_41948 | Besides the body of water by the city where Erwin Speckler died, and the Irish Channel, what else was lowered in the last cold phase? | English Channel | sea level | 0.000 | 0.000 | 52.271 |
| 3hop1__379315_160713_77246 | What is the Arabic dictionary meaning of the word referring to the majority religion in the area of British India, that became India when the country containing Barawal was formed? | the country of India | Hindu | 0.000 | 0.000 | 13.152 |
| 3hop2__57238_1926_54362 | Who burned down the city where Keeping Up with the Joneses is set, during the war marking the first time that the number of black people living in NYC declined? | Confederate Gen. John Bell Hood | Richmond | 0.000 | 0.000 | 57.214 |
| 2hop__853227_68633 | Arthur Marryatt is a member of a sports association that has who as president? | Thomas Bach | Sir Clive Woodward | 0.000 | 0.000 | 12.697 |
| 3hop1__19768_19788_15107 | What percentage of Eritrea is estimated to adhere to the religion widely practiced in the region with the 2nd largest rain-forest in the world? | 48% | 50% | 0.000 | 0.000 | 49.105 |
| 2hop__35173_17335 | When did the person who ended the Archaemenid Empire by conquest in 330 BC die? | 323 BC | 323 BC | 1.000 | 1.000 | 7.754 |
| 3hop1__632623_160713_77246 | In the arabic dictionary what is the meaning of the word that also refers to the majority religion in India during the creation of the country where Rabwah is located? | the country of India | Hindu | 0.000 | 0.000 | 57.463 |
| 2hop__85993_512508 | What was the person who discovered that nuclei of atoms contained neutrons with no charge a participant in? | Manhattan Project | Manhattan Project | 1.000 | 1.000 | 49.344 |
| 3hop2__792915_467331_162182 | In what region of the country containing the Cái River is the city that has the Zone 5 Military Museum located? | South Central Coast | Red River Delta | 0.000 | 0.000 | 7.558 |
| 2hop__159215_779396 | Where was the person who wrote about the rioting being a dividing factor in Birmingham educated? | University of Glasgow | Morehouse College | 0.000 | 0.000 | 6.756 |
| 3hop1__389955_132457_47686 | Where do Greyhound buses leave from in the city where the band that recorded the album I'm Alive and on Fire formed? | Toronto Coach Terminal | Salt Lake City International Airport | 0.000 | 0.000 | 24.972 |
| 2hop__386236_350892 | Where was Osbert Sitwell's sibling educated? | Eton College | Eton College | 1.000 | 1.000 | 6.567 |
| 2hop__158275_35739 | When was one internet browser's version of Windows 8 made accessible? | September 2013 | 2012 | 0.000 | 0.000 | 15.167 |
| 2hop__280583_229757 | Who is the spouse of the creator of The Mighty B? | Will Arnett | Lisa | 0.000 | 0.000 | 28.082 |
| 3hop2__127483_84553_10557 | What was the language of Auctor of who built a european empire and was crowned emperor of the romans in 800 era later known as? | Medieval Latin | Latin | 0.000 | 0.667 | 29.291 |
| 2hop__6744_6731 | What two dialects are included in the other pattern other vowels follow? | Northwestern Catalan and Valencian | Hawaiian Samoan | 0.000 | 0.000 | 19.297 |
| 3hop1__617062_127905_86588 | What is the longest home run in the history of the league that Jim Wilson's team is a part of? | 582 feet (177 m) | 565 feet | 0.000 | 0.333 | 38.478 |
| 3hop2__132957_132798_40768 | When did the make of Acura Legend, the company that made toyopet Master and Nissan open US assembly plants? | 1981 | 1986 | 0.000 | 0.000 | 36.310 |
| 3hop1__144678_443779_52195 | The Commission of Truth and Friendship established governments for Bustami's birth country and a second country that had who as president after declaring independence? | Francisco Guterres | Islam Karimov | 0.000 | 0.000 | 38.019 |
| 2hop__362912_97805 | Which was did the operator of the Jurmo class landing craft serve in? | Winter War | Finnish Navy | 0.000 | 0.000 | 20.459 |
| 4hop3__754156_88460_30152_20999 | How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country? | The dynasty regrouped and defeated the Portuguese | military conquest | 0.000 | 0.000 | 28.132 |
| 2hop__62823_654855 | What political party was the candidate who garnered a million votes during the 1920 election despite being in prison, a member of? | Socialist Party of America | Socialist Party | 0.000 | 0.667 | 14.314 |
| 2hop__742831_156034 | Of which river is the river Kichi Sipi Bridge crosses a tributary? | Hudson's Bay | Winnipeg River | 0.000 | 0.000 | 12.524 |
| 2hop__510647_507085 | Who is the father of the person that Crockett County is named after? | John Crockett | Nathaniel Crockett | 0.000 | 0.500 | 14.137 |
| 4hop1__525129_315334_131926_87157 | What is the direction of flow of the river by the city where the Lots More Blues, Rags and Hollers performer was formed? | rises in northern Minnesota and meanders slowly southwards | south | 0.000 | 0.000 | 24.342 |
| 2hop__35445_22458 | What is the full name of the organization that publishes he Laws of the Game? | International Federation of Association Football | International Football Association Board | 0.000 | 0.667 | 4.248 |
| 3hop1__131820_59747_90139 | Which star of the Real Housewives show, set in the largest city by population in the state where WEKL is located, is married to an NFL player? | Kim Zolciak | Kim Kardashian | 0.000 | 0.500 | 60.193 |
| 2hop__38828_89399 | Who voices the character in Spongebob Squarepants who is named after a glowing species found in some beaches in Portugal? | Mr. Lawrence | Carolyn Lawrence | 0.000 | 0.500 | 18.204 |
| 2hop__77518_548781 | Where was the president born on the fourth of July born? | Plymouth Notch | Vermont | 0.000 | 0.000 | 15.872 |
| 3hop1__135794_87694_64412 | When did the city where the Basilica that is named after the same saint as the one in San Pietro in Montorio located in become a country? | 11 February 1929 | 1929 | 0.000 | 0.500 | 32.951 |
| 2hop__13574_13498 | How many UEFA Super Cup awards have been received by the team that has won the treble competitions twice? | five | 2 | 0.000 | 0.000 | 19.815 |
| 2hop__757497_78606 | In Grown Ups, who plays the wife of the actor who produced The Chosen One? | Joyce Van Patten | Jennifer Aniston | 0.000 | 0.000 | 57.666 |
| 3hop1__857_846_7791 | How many parties rule the city where the Yongle emperor greeted the person to whom the edict was addressed? | one-party | one | 0.000 | 0.000 | 42.421 |
| 2hop__131275_72870 | Who did the state where Christian Science Pleasant View Home was located vote for in 2016? | Hillary Clinton | Donald Trump | 0.000 | 0.000 | 12.193 |
| 2hop__95878_88628 | Who did the actor in Advernture's End play in True Grit? | U.S. Marshal Rooster Cogburn | Matt Damon | 0.000 | 0.000 | 69.251 |
| 2hop__40970_90367 | What was the role of God in the Biblical crossing of the sea where herbivorous fishes have been seen feeding on gelatinous zooplankton? | the Red Sea was parted by God | God parted the sea to allow the Israelites to cross. | 0.000 | 0.429 | 14.206 |
| 2hop__122651_141338 | What company is Zena Tsarfin's school a part of? | City University of New York | Broad Institute | 0.000 | 0.000 | 22.155 |
| 2hop__215124_83837 | Who sings Home Alone Tonight with the Rain is a Good Thing singer? | Karen Fairchild | Danny O'Donoghue | 0.000 | 0.000 | 55.751 |
| 2hop__79875_72187 | What character in A League of their Own was played by the actress who played Thelma in the movie Thelma and Louise? | Dottie Hinson | Dottie Hinson | 1.000 | 1.000 | 9.623 |
| 2hop__283298_22402 | How many students attend the university employing Marie Theres Fögen? | nearly 25,000 | 90000 | 0.000 | 0.000 | 13.372 |
| 2hop__384238_74735 | What song did Paul McCartney write for Sean Lennon's siblings? | ``Hey Jude '' | Julian | 0.000 | 0.000 | 15.206 |
| 2hop__124933_653666 | Who is the mother of the father of Julia Murdock Smith? | Lucy Mack Smith | Insufficient information | 0.000 | 0.000 | 12.001 |
| 2hop__170823_120171 | What year did the publisher of Labyrinth end? | 1986 | 1989 | 0.000 | 0.000 | 26.286 |
| 3hop1__132795_40769_64047 | When did the luxury division of the company that produces TMG EV P001 change the body style of rx 350? | Sales began worldwide in April 2012 | 2014 | 0.000 | 0.000 | 18.377 |
| 4hop2__9988_685934_70784_61381 | When was the country established that lies north of the region that prevailed with the disgrace of the "Near East" and the body of water containing The World islands? | 1932 | 1923 | 0.000 | 0.000 | 68.062 |
| 2hop__411091_625230 | In which county is Mark Dismore's birthplace located? | Hancock County | Cornwall | 0.000 | 0.000 | 8.349 |
| 2hop__642686_7292 | Along with Kenny G and the performer of Hello Tomorrow, what artist was featured on Smooth Jazz Stations? | George Benson | The Four Seasons | 0.000 | 0.000 | 50.127 |
| 2hop__21991_43945 | Why was Common Sense, written by the same author as The Rights of Man, an important work? | crystallized the rebellious demand for independence from Great Britain | argued for American independence | 0.000 | 0.333 | 25.105 |
| 2hop__835710_7298 | Along with the Closer performer, what notable pop artist started out his career on adult contemporary radio? | Michael Bublé | John Mayer | 0.000 | 0.000 | 31.419 |
| 2hop__91248_84207 | Who was the last soldier of the conflict that saw the government getting involved in healthcare to die? | Albert Henry Woolson | No known soldier | 0.000 | 0.000 | 37.628 |
| 2hop__7906_78851 | Which professional sports team would you not see play a home game where Jim Foster came up with the idea for arena football? | Brooklyn Nets NBA |  | 0.000 | 0.000 | 75.324 |
| 2hop__607517_161450 | Where is the province containing Bandar-e Gaz County located? | in the north-east of the country south of the Caspian Sea | Hormozgan Province | 0.000 | 0.000 | 5.028 |
| 2hop__123509_5385 | How many people work in the university that is related with Mark F. Green? | 11,900 | 30000 | 0.000 | 0.000 | 10.851 |
| 3hop1__603558_87694_124169 | When was the end of the governorship where the head of Catholicism is, along with the basilica named after the saint who San Pietro in Ciel d'Oro is named for? | 1952 | 1216 | 0.000 | 0.000 | 61.124 |
| 2hop__77878_189357 | Where was the first woman appointed to the Supreme Court educated? | Stanford University | Stanford University | 1.000 | 1.000 | 14.067 |
| 3hop1__408237_400692_51423 | What is the name of the castle in the birthplace of Harmony's performer? | Casa Loma | Blenheim Palace | 0.000 | 0.000 | 28.919 |
| 3hop1__5544_34179_23375 | What year saw the publication of the version of the Molotov–Ribbentrop Pact established by the man who succeeded the leader of the group which created the Soviet state? | 1948 | 1939 | 0.000 | 0.000 | 12.524 |
| 3hop1__127716_157828_239539 | Where is the headquarter of the Radio Television of the country whose co-official language is the same as the one Politika is written in? | Pristina | Belgrade | 0.000 | 0.000 | 19.674 |
| 3hop1__92251_34179_23375 | What year saw the publication of the version of the Molotov–Ribbentrop Pact, engineered by the politician who succeeded the leader of the Bolsheviks in the November Revolution? | 1948 | 1939 | 0.000 | 0.000 | 17.466 |
| 2hop__57638_88526 | Who was picked in the NBA draft before the player with the highest point average in NBA history? | Greg Oden | Kareem Abdul-Jabbar | 0.000 | 0.000 | 72.244 |
| 4hop1__46196_497223_15840_36014 | What was Nintendo's limit on games per developer per year for the platform supporting the video game named after the sports organisation in charge of the Super Bowl halftime show? | five | 10 | 0.000 | 0.000 | 25.076 |
| 2hop__433694_20273 | In what year did Margaret Knox's spouse pass away? | 1572 | 1572 | 1.000 | 1.000 | 19.245 |
| 2hop__415518_616002 | What award did the author of The Happy Journey to Trenton and Camden receive? | Pulitzer Prize for Drama | Pulitzer Prize | 0.000 | 0.667 | 55.360 |
| 4hop3__601548_836463_161616_77103 | When did the birthplace of the Live and Beyond performer become the capitol of the state Knowles was from? | 1839 | 1825 | 0.000 | 0.000 | 34.444 |
| 4hop3__373866_5189_38229_86687 | A group of nations of which the People's Socialist Republic of Albania was a member was controlled by one country. When did that country agree to a unified Germany inside of the organization Eisenhower would not discuss during the campaign? | May 1990 | 1955 | 0.000 | 0.000 | 25.194 |
| 3hop1__604991_339990_54675 | What mountain can you see from Portland in the state where Frozen Run's county is located? | Tualatin Mountains | Mount Tabor | 0.000 | 0.000 | 33.339 |
| 2hop__80375_90181 | what is the deepest part of the ocean off the east coast of the united states? | Milwaukee Deep | Milwaukee Deep | 1.000 | 1.000 | 31.647 |
| 2hop__9223_9237 | When did the first mosque open in the place where Tobolar Copra plant is located? | September 2012 | 1520 | 0.000 | 0.000 | 26.958 |
| 3hop2__304722_591230_63959 | When was the last time Ken Faulkner's team beat the 1894-95 FA Cup winner? | 1 December 2010 | 1932 | 0.000 | 0.000 | 37.491 |
| 2hop__481349_302087 | Which company owns the manufacturer of Learjet 60? | Bombardier Inc. | Bombardier | 0.000 | 0.667 | 10.258 |
| 3hop2__304722_297619_63959 | When was the last time Cian Hughton's team beat the 1894–95 FA Cup winner? | 1 December 2010 | January 15 2022 | 0.000 | 0.000 | 15.775 |
| 2hop__159741_40169 | When did a storm disperse the fleet led by the English king who trusted William de Longchamp? | 1191 | 1213 | 0.000 | 0.000 | 10.536 |
| 2hop__151750_141308 | What company is the record label of Magic Christian Music part of? | Apple Corps | Universal Music Group | 0.000 | 0.000 | 24.916 |
| 2hop__43579_289940 | What sports team is the person who had the most sacks in the NFL this season a member of? | New England Patriots | Cleveland Browns | 0.000 | 0.000 | 16.571 |
| 4hop3__316459_41402_599212_13584 | A country gained control of Florida after the war in which Charles Edward Nugent saw action. Besides being in different parts of that country, what other differences are there between Real Madrid and Hostafrancs's city's team? | two cities | War of 1812   Real Madrid is Spanish, Hostafrancs's team is Belgian | 0.000 | 0.000 | 65.119 |
| 4hop1__145422_698949_157828_162309 | When did the country, that has the original language of the film named after Vladimir Karalić's birthplace as a co-official language, first attend the Olympics as an independent team? | 2016 | 1992 | 0.000 | 0.000 | 15.655 |
| 3hop1__127821_157828_162309 | When did the country using Olivera Marković's native language as one of its official languages first have its own Olympic team? | 2016 | 1912 | 0.000 | 0.000 | 51.463 |
| 2hop__21104_16335 | The group representing at least 70% of Jews worldwide and Italians may be genetically similar due to what two factors? | inter-marriage and conversions in the time of the Roman Empire | shared ancestry and intermarriage | 0.000 | 0.333 | 22.608 |
| 2hop__543853_124498 | Who is the father of the Labo M performer? | Louis Chedid | Masaru Kato | 0.000 | 0.000 | 7.693 |
| 2hop__728969_131890 | Which is the body of water near George Mills' place of birth? | River Thames | Waikato River | 0.000 | 0.500 | 13.781 |
| 3hop2__103871_20335_9331 | What dissolved the privileges in Poland of the empire where Tamar Abakelia was from, the Kingdom that acquired some Thuringian territory in 1815, and the Habsburg Monarchy? | March Constitution of Poland | Congress of Vienna | 0.000 | 0.286 | 50.446 |
| 3hop1__765565_326948_86925 | What is the symbols of the Saints from the place where Here Come the Girl's performer was born called? | fleur - de-lis | flame | 0.000 | 0.000 | 28.113 |
| 2hop__2416_86208 | When does the river that pumps the engine flood every year? | August | every year | 0.000 | 0.000 | 28.747 |
| 2hop__421324_97805 | Which war did the operator of Finnish coastal defence ship Väinämöinen serve in? | Winter War | World War II | 0.000 | 0.400 | 18.292 |
| 2hop__56851_343058 | Who is the spouse of the actress who played the fairy god mother in Shrek 2? | Adrian Edmondson | David Niven | 0.000 | 0.000 | 32.369 |
| 2hop__249867_557232 | Which country is the Desert Forest Golf Club located in? | Maricopa County | United States | 0.000 | 0.000 | 22.841 |
| 4hop1__391525_49925_13759_736921 | Where is the district that the person who wanted to reform and address John Kodwo Amissah's religion preached a sermon on Marian devotion before his death located? | Saxony-Anhalt | Nsawam | 0.000 | 0.000 | 32.607 |
| 4hop2__9988_158985_70784_79935 | When was the kingdom lying immediately to the north of the region prevailing with "Near East" disgrace and the region having the most growth in desalination for agricultural use created? | 1930 | 1921 | 0.000 | 0.000 | 33.332 |
| 3hop1__558449_503371_21711 | When was the Palau de la Generalitat built in the city in which died the man called Martin of the region of Spain where Malanquilla is located? | built in the 15th century | 1270 | 0.000 | 0.000 | 15.830 |
| 4hop3__531933_836463_161616_77103 | When did the city where Souvenir's performer was born become the capitol of the state Knowles was from? | 1839 | 1817 | 0.000 | 0.000 | 36.919 |
| 3hop1__280834_547811_41132 | How many times did plague occur in the place where Bajazet's composer was born? | 22 | two | 0.000 | 0.000 | 30.995 |
| 2hop__17827_54024 | Which king sold part of the Crown Freehold and commissioned a bible translation? | King James I of England | James I | 0.000 | 0.571 | 19.787 |
| 2hop__765678_310556 | Who was married to the star of No Escape? | John Profumo | Sarah Jones | 0.000 | 0.000 | 21.030 |
| 2hop__635187_861533 | What province shares a border with the province where Lago District is located? | Cabo Delgado Province | Lanao del Sur | 0.000 | 0.000 | 13.039 |
| 4hop2__71753_158279_70784_61381 | When was the region immediately north of the region where Israel is located and the final destination for the export crops established? | 1932 | Golan Heights | 0.000 | 0.000 | 41.418 |
| 3hop1__324448_831637_91775 | What does SEAL stand for in the military branch that operates USS Prevail? | Sea, Air, and Land | United States Navy | 0.000 | 0.000 | 32.304 |
| 2hop__59048_827343 | Whose sister played Susie in miracle on 34th street? | Lana Wood |  | 0.000 | 0.000 | 78.950 |
| 2hop__53175_548781 | What is the birth place of the US president with whose election the republican ascendancy of the 1920's began? | Plymouth Notch | Blooming Grove, Ohio | 0.000 | 0.000 | 11.230 |
| 3hop1__239036_15840_36002 | What were the Genesis's advantages over the platform of Xexys? | built on 16-bit architectures and offered improved graphics and sound | higher leverage and better security | 0.000 | 0.133 | 21.109 |
| 2hop__131951_643670 | What is the name for the mouth of the watercourse of the body of water by Rotterdam Centrum? | Het Scheur | Maasmond | 0.000 | 0.000 | 13.617 |
| 4hop2__105527_14670_8987_8529 | A country's military branch, the equivalent of which in the US contains the Air Defense Artillery, was unprepared for the invasion of the country that developed the ZH-29. When was the word "Slavs" used in the national anthem of the unprepared country? | 1943–1992 |  | 0.000 | 0.000 | 80.792 |
| 3hop1__142044_538202_84283 | Who is the owner of the record label of the performer responsible for the album Just Got Started Lovin' You? | Warner Music Group | Universal Music Group | 0.000 | 0.667 | 64.175 |
| 2hop__763380_74735 | What song did Paul McCartney write for the person who performed Too Late for Goodbyes? | ``Hey Jude '' | For You I'll Wait | 0.000 | 0.000 | 69.528 |
| 2hop__159816_37168 | According to QS World University Rankings where does the university where Hawass graduated rank? | 551-600 | 450 | 0.000 | 0.000 | 19.007 |
| 2hop__56806_7298 | What pop artist started his career on adult contemporary radio along with the subject of the song The One That Got Away? | Michael Bublé | Kelly Clarkson | 0.000 | 0.000 | 54.860 |
| 4hop1__72247_497223_15840_36014 | There is a video game named after the sports association that chooses where the super bowl is held. What was Nintendo's limit on games per developer per year on that game's platform? | five | 3 | 0.000 | 0.000 | 14.549 |
| 4hop2__9988_261673_70784_61381 | The country in what was the Near East and bordering the sea over which shamal winds blow was established when? | 1932 | 1961 | 0.000 | 0.000 | 45.670 |
| 2hop__71871_494136 | Who is the spouse of the actor playing Bart's father in I can only imagine? | Meg Ryan | Karen Varney | 0.000 | 0.000 | 48.561 |
| 3hop1__641921_42197_18397 | Where did the arguer that Nadezhda Tkachenko's country of citizenship had become an imperialist power declare he would intervene in the Korean conflict? | the Politburo | United Nations | 0.000 | 0.000 | 37.926 |
| 4hop1__436202_765799_282674_759393 | What is the seat of the county sharing a border with the county in which J. P. Hayes was born? | Green Bay | White Plains | 0.000 | 0.000 | 18.957 |
| 2hop__58111_7298 | Who started his career on adult contemporary radio along with the person who plays andy bernard's brother on the office? | Michael Bublé | Brian Posehn | 0.000 | 0.000 | 48.651 |
| 4hop1__455862_508773_85832_745702 | Whose navigator father explored the east coast of the region where Ignacio Esparza was later born? | Sebastian Cabot | Ferdinand Magellan | 0.000 | 0.000 | 73.705 |
| 3hop1__103881_443779_52195 | Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country that Tony Gunawan is from? | Francisco Guterres | Francisco Guterres | 1.000 | 1.000 | 35.065 |
| 2hop__252311_366220 | Who founded the company that distributed the film UHF? | Mike Medavoy | Robert L. Johnson | 0.000 | 0.000 | 25.720 |
| 2hop__149862_108549 | Who created the work that features the appearance of Cosmic Boy? | Alfred Gough | Robert Bernstein and Mike Sekowsky | 0.000 | 0.000 | 12.427 |
| 4hop1__473563_309800_355213_203322 | Which county shares border with the county having the birth city of the director of Let the Good Times Roll? | Cabarrus County | Santa Clara | 0.000 | 0.000 | 32.683 |
| 4hop3__547073_88460_30152_20999 | How were people from whom the new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and Si Phan Don's country? | The dynasty regrouped and defeated the Portuguese | no record | 0.000 | 0.000 | 27.282 |
| 2hop__156700_63853 | What is the source of the river of which Kraai River is a tributary? | Thaba Putsoa | Orange River's source is in the Maluti Mountains of Lesotho. | 0.000 | 0.000 | 16.120 |
| 3hop1__320697_547811_80702 | What is the name of the famous bridge located in the birthplace of the composer of Nulla in mundo pax sincera? | Rialto Bridge | Beethoven Bridge | 0.000 | 0.500 | 9.896 |
| 2hop__23459_35123 | What was the lifespan of the most influential figure in Islamic philosophy? | 980–1037 | 980–1037 | 1.000 | 1.000 | 6.745 |
| 2hop__29893_29905 | How many mandatory transmitters of the owner and operator of most CBC TV stations were updated before the deadline? | only about half | 12 | 0.000 | 0.000 | 11.524 |
| 2hop__825727_584042 | What league includes the sports team that has Maycon Carvalho Inez? | K-League | Campeonato Brasileiro Série A | 0.000 | 0.000 | 16.530 |
| 3hop1__351696_42197_18397 | Where did the person who argued that the country where Ural Oblast is located had become an imperialist power declare he would intervene in the Korean conflict? | the Politburo | Moscow | 0.000 | 0.000 | 45.756 |
| 2hop__160223_58067 | Who was given India's highest civilian honor before he became President of India? | A.P.J. Abdul Kalam | Dr. Sarvepalli Radhakrishnan | 0.000 | 0.000 | 30.043 |
| 2hop__706637_93362 | What is the movie starring Preity Zinta, Madhuri Dixit and the spouse of Kajol? | Yeh Raaste Hain Pyaar Ke | Dilwale Dulhania Le Jayenge | 0.000 | 0.000 | 44.558 |
| 2hop__2683_577502 | Who married the governor who passed a law in honor of Donda West? | Maria Shriver | Maggie Pritzker | 0.000 | 0.000 | 26.023 |
| 2hop__595747_31270 | How many miles is Sloan Thomas' birthplace from Nashville? | 45 | 85 | 0.000 | 0.000 | 46.135 |
| 4hop2__105527_14670_8987_8974 | Country A developed the ZH-29 and invaded Country B because their military branch that the Air Defense Artillery is a branch of was unprepared. Country B was the only communist country to have an embassy where? | Alfredo Stroessner's Paraguay | Cuba | 0.000 | 0.000 | 56.872 |
| 2hop__204146_850295 | Who is the sibling of the author of Kapalkundala? | Sanjib Chandra | Rajanikanta Chattopadhyay | 0.000 | 0.000 | 10.004 |
| 2hop__388523_75487 | Who is the federal leader of the party that James Rajotte is a member of? | Andrew Scheer | Pierre Poilievre | 0.000 | 0.000 | 29.012 |
| 4hop3__131775_29339_508306_70744 | Which has more national championships, university of the state having Fort Hill or university of the state whose primary was won by Edwards besides the state having Ted Wright Stadium? | University of South Carolina | University of South Carolina | 1.000 | 1.000 | 66.080 |
| 4hop1__88342_49853_128008_46894 | Who has the lowest batting average in the league that the team that has won the most titles from the competition they give out MVP awards plays for? | Bill Bergen | no data | 0.000 | 0.000 | 70.665 |
| 3hop1__426806_42197_18397 | Where did who argued that the country of citizenship of Victor Denisov had itself beome an imperialist power declare that he would intervene in the Korean conflict? | the Politburo | the United Nations | 0.000 | 0.000 | 48.414 |
| 2hop__695432_18803 | The operator of Brent field announced it was in the process of doing what in April 2010? | trying to find a potential buyer for all of its operations in Finland | decommissioning | 0.000 | 0.000 | 42.294 |
| 2hop__800166_156034 | What is the river where Long Spruce Generating Station is located a tributary of? | Hudson's Bay | James Bay | 0.000 | 0.500 | 19.436 |
| 3hop1__354480_834494_34053 | What was the wettest year for the city within the county that houses Eric Marcus Municipal Airport? | 1905 | 1998 | 0.000 | 0.000 | 26.970 |
| 2hop__78873_461854 | Who is the female star in Gone with the Wind married to? | Laurence Olivier | Rhett Butler | 0.000 | 0.000 | 39.207 |
| 4hop1__324759_378185_282674_759393 | What is the capital of the county that shares a border with the county where Pierce Manufacturing's headquarters are located? | Green Bay | Trinidad | 0.000 | 0.000 | 19.439 |
| 4hop1__151650_5274_458768_33633 | What date did Gaspar reach the region which would be the headquarters of the record label of which the only larger label is the one that signed Desde El Principio? | August 3, 1769 | 2002 | 0.000 | 0.000 | 24.955 |
| 3hop2__87184_90327_76291 | When did the majority party in the House of Representatives gain control of the body which approves members of the Cabinet? | January 2015 | 1932 | 0.000 | 0.000 | 58.576 |
| 2hop__6827_55848 | When was the first branch of the company that bought FleetBoston Financial established? | November 3, 1930 | 1904 | 0.000 | 0.000 | 28.682 |
| 3hop2__25177_19506_23299 | After the period which limited the supply of shellac; Białystok, Przemyśl were returned to what country along with the region they had been given in the Treaty of Schönbrunn? | Polish state | Austria | 0.000 | 0.000 | 57.240 |
| 2hop__68162_482364 | Who was the father of the leader of the Argonauts in the search for the golden fleece? | Aeson | Aeetes | 0.000 | 0.000 | 13.587 |
| 2hop__44259_747874 | Who is the sibling of the player who won the 2012 U.S. Open Men's Singles final? | Jamie Murray | Stefan Djokovic | 0.000 | 0.000 | 6.006 |
| 2hop__5190_64006 | By approximately what percentage did the Black Death reduce the population of the region that the Soviet Union feared invasion from? | 30% to 65% | 30-60% | 0.000 | 0.000 | 52.256 |
| 2hop__95673_121319 | Where did the person featured in the film Sammy Going South study or work? | American Academy of Dramatic Arts | University of California, Los Angeles | 0.000 | 0.200 | 57.775 |
| 2hop__364489_861485 | Where is the location of the headquarters of the publisher of Bamboo Among the Oaks? | Minnesota History Center | Madison, Wisconsin | 0.000 | 0.000 | 9.137 |
| 2hop__25797_990 | Which iPod generation introduced compatibility with the industry standard connection that was first widely used at version 1.1? | third generation | iPod Nano | 0.000 | 0.000 | 13.480 |
| 3hop1__664835_650651_7262 | Who was a prominent figure in the radio division of the network of the version of The Biggest Loser: The Final Chance, set in the country where Bokok is located? | Walter Sabo | Sokha Chum | 0.000 | 0.000 | 31.541 |
| 3hop1__497845_629431_124169 | On what date did the Governor of the place of death of Princeps Pastorum's author end? | 1952 | August 20 1914 | 0.000 | 0.000 | 75.466 |
| 2hop__35566_17873 | What happened to Buckingham after the departure of the queen who Martha Ann Ricks presented the famed Liberian coffee tree to? | palace was seldom used, even neglected | Buckingham Palace became the residence of King Charles III. | 0.000 | 0.143 | 8.417 |
| 2hop__57638_72467 | What team picked the highest point average holder in NBA history in the NBA draft? | Seattle SuperSonics | Milwaukee Bucks | 0.000 | 0.000 | 8.000 |
| 2hop__818302_25719 | At the end of which year did the tribes from the location of the Lusatia invade the Roman Empire? | 406 | 180 | 0.000 | 0.000 | 42.485 |
| 2hop__256336_714772 | Who is the spouse of a cast member from King of the Mountain? | Lisa Rinna | Rick Springfield | 0.000 | 0.000 | 17.517 |
| 4hop3__661103_698586_1926_54362 | Who burned down the city where Dunn Dunn's artist died in the war after which black people declined living in NYC? | Confederate Gen. John Bell Hood |  | 0.000 | 0.000 | 78.946 |
| 3hop1__132968_40769_64047 | When did the luxury division of the company that made Daihatsu Charmant change the body style of the rx 350? | Sales began worldwide in April 2012 | 2007 | 0.000 | 0.000 | 33.145 |
| 2hop__123104_77615 | When did the military instruction start in the university that Paulo Manalo attended? | 1912 | 1902 | 0.000 | 0.000 | 28.617 |
| 2hop__25788_25839 | What is the maximum load drawn by the device that was designed to standardize connection of computer peripherals? | five unit loads | 2.5 watts | 0.000 | 0.000 | 27.999 |
| 2hop__41158_35124 | How many books were said to have been written by the person who proposed explanations for the origins of earthquakes and the formation of mountains? | 450 | one | 0.000 | 0.000 | 38.382 |
| 3hop1__320697_547811_41132 | How many times did plague occur in the birth city of the composer of Nulla in mundo pax sincera? | 22 | once | 0.000 | 0.000 | 29.275 |
| 4hop2__161602_474028_88460_21034 | What year did the country that is the natural boundary between where the tournament was hosted and where A Don is located host the ASEAN Conference? | hosted the summit in 2014 | 2012 | 0.000 | 0.000 | 20.850 |
| 3hop1__764280_834494_34053 | What was the wettest year of the administrative territorial entity that contains the administrative territorial entity where Drexel Heights is located? | 1905 | 1972 | 0.000 | 0.000 | 18.092 |
| 3hop2__89854_90327_76291 | When did the party that hold the majority in the House of Reps take control of the branch that approves members of the American cabinet? | January 2015 | 1932 | 0.000 | 0.000 | 68.701 |
| 3hop2__565959_223623_162182 | In what region of the country containing Mount Can is the birthplace of John Phan located? | South Central Coast | Central Highlands | 0.000 | 0.400 | 11.923 |
| 2hop__684839_139312 | What was the cast member from Real Life nominated for? | Academy Award for Best Supporting Actor | Lisa Lopez Teen Choice Award | 0.000 | 0.182 | 40.218 |
| 2hop__95172_456836 | Where was the person who acted in Signora Enrica born? | La Goulette | Rome | 0.000 | 0.000 | 20.635 |
| 2hop__452580_70131 | Who constituted the free crops in the state having Free State of Brunswick? | consisting largely of World War I veterans | peasants | 0.000 | 0.000 | 30.828 |
| 3hop1__23058_20608_36327 | What did Bloomberg name the country formerly known as the region where the Spanish and Portuguese took most of their slaves from? | top emerging market economy in Africa | The Gambia | 0.000 | 0.000 | 24.270 |
| 2hop__731228_609866 | What district is the headquarter of Julia's House located? | East Dorset | Camden | 0.000 | 0.000 | 10.693 |
| 3hop2__851134_326964_7713 | How long had Pfrang Association's headquarters location been the capitol city of Yaxing Coach's headquarters location? | about 400 years | unknown | 0.000 | 0.000 | 20.541 |
| 3hop1__626214_503371_21711 | 3 When was the Palau de la Generalitat in the place of death of martin of the administrative territorial entity Navardun is located constructed? | built in the 15th century | 1270 | 0.000 | 0.000 | 36.441 |
| 4hop1__152562_5274_458768_33637 | How many ethnic minorities were looked at differently in the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located? | two | 12 | 0.000 | 0.000 | 58.721 |
| 2hop__645448_77615 | When did military instruction start at the place where Larry Alcala was educated? | 1912 | 1908 | 0.000 | 0.000 | 29.486 |
| 2hop__250030_7298 | Who started out his career on adult contemporary radio along with the performer of All That Echoes? | Michael Bublé | James Taylor | 0.000 | 0.000 | 50.014 |
| 2hop__78766_461854 | Who is the spouse of the actor who played Scarlet in Gone with the wind? | Laurence Olivier | Laurence Olivier | 1.000 | 1.000 | 6.098 |
| 2hop__135653_669373 | Who is the employer of the person Mach's principle was named after? | Charles University | University of Vienna | 0.000 | 0.400 | 8.950 |
| 3hop1__857_846_7795 | In 2011, what was the estimated population of the city where the Yongle emperor greeted the person to whom the edict was addressed? | 8.11 million | 19.6 million | 0.000 | 0.500 | 55.857 |
| 3hop1__406043_831637_91775 | What does seal stand for in the seals of the service operating list of destroyer classes of the operator of USS Peril? | Sea, Air, and Land | Seal | 0.000 | 0.000 | 8.426 |
| 3hop1__465684_160545_62931 | Where in the country where Bodindecha was born was The Beach filmed? | island Koh Phi Phi | Koh Phi Phi | 0.000 | 0.857 | 5.575 |
| 3hop1__346416_831637_91775 | What does seal stand for in the U.S. military branch that operates a list of destroyer classes, and also operates NC? | Sea, Air, and Land | Navy | 0.000 | 0.000 | 21.178 |
| 3hop1__865037_214799_259594 | Border troops of the country of literature of the country of citizenship of Rainer Ernst are from what country? | GDR | Germany | 0.000 | 0.000 | 41.813 |
| 2hop__136214_483189 | The person whom Sikyona is named after is part of what? | Greek mythology | Katipunan | 0.000 | 0.000 | 30.809 |
| 4hop1__131113_59747_211319_557671 | Which county shares a border with the county where the most populous city in the state where Washington State Prison can be found is located? | Presque Isle County | Pierce County | 0.000 | 0.400 | 22.686 |
| 2hop__147212_60904 | Who owns the rehabilitation center at the place where Hank Moonjean died? | Jack Michel | Dr. Melfi | 0.000 | 0.000 | 24.860 |
| 2hop__6956_192417 | The league the New England Patriots belongs to is an example of what? | sports league | NFL | 0.000 | 0.000 | 27.221 |
| 3hop1__587540_514761_594810 | What record label does the spouse of Mama's Broken Heart's performer belong to? | Warner Bros. Records | Warner Bros. Records | 1.000 | 1.000 | 10.635 |
| 3hop2__88342_93066_90766 | When was the last time the team that won the American League East in 2017 and the Dodgers met in the championship event that precedes the MLB MVP award? | 1981 |  | 0.000 | 0.000 | 70.363 |
| 3hop1__57186_237521_291682 | Who's married to the man who produced a documentary about the singer of She's Out of My Life? | Liza Minnelli |  | 0.000 | 0.000 | 69.800 |
| 2hop__61318_362941 | What is the place of birth of the person who submitted the first version of the equal rights amendment to congress in 1923? | Mount Laurel Township | New Jersey | 0.000 | 0.000 | 23.855 |
| 2hop__13539_13493 | In 2009, what series of wins did the team who won the first match in the 1943 Copa del Generalissimo achieve? | continental treble | 2009 Champions League | 0.000 | 0.000 | 27.262 |
| 2hop__699172_141338 | What company is the school where Lawrence Landweber was educated a part of? | City University of New York | University of Chicago | 0.000 | 0.500 | 45.402 |
| 2hop__57638_90991 | Who was picked over the player with the highest point average in NBA history in the draft? | Greg Oden | Kareem Abdul-Jabbar | 0.000 | 0.000 | 33.674 |
| 3hop1__446612_160545_62931 | Where was the film The Beach filmed in the country where Ercole Manfredi died? | island Koh Phi Phi | Thailand | 0.000 | 0.000 | 39.348 |
| 4hop1__152562_5274_458768_33633 | When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located? | August 3, 1769 | 1498 | 0.000 | 0.000 | 27.654 |
| 2hop__345_59409 | Who was the person who purchased Tidal's parent company in 2015 talking about in the song 'Cry?' | three different relationships he had in the past | ex | 0.000 | 0.000 | 47.068 |
| 3hop2__30956_65244_84681 | When did the people who controlled New York before it became an English colony, come to the country where John Kerry's yacht was built? | 13 December 1642 | 1626 | 0.000 | 0.000 | 39.795 |
| 2hop__75207_365216 | In which province is San Clemente, from the country where Fuser and Alberto meet the indigenous couple who were traveling to look for work? | Talca Province | Baja California | 0.000 | 0.000 | 28.527 |
| 2hop__29349_92763 | What is the enrollment where the first debate between Kerry and Bush took place? | 16,801 | Boston University | 0.000 | 0.000 | 7.795 |
| 2hop__629030_56873 | When did the 1979-80 European Cup winner win the FA Cup? | 1898 and 1959 | 1980 | 0.000 | 0.000 | 13.287 |
| 3hop1__139773_88110_77129 | What was the first OS to support the Hyper-V protocol, developed by the company that the letters MS stand for, in the network where The Rachel Maddow Show is found? | Windows 8 | Windows Server 2008 R2 | 0.000 | 0.333 | 25.955 |
| 3hop2__132957_379231_40768 | When did Nissan, the Acura Legend maker and the Scion xD manufacturer open US assembly plants? | 1981 | 1984 and 1989 | 0.000 | 0.000 | 20.539 |
| 4hop3__524186_219173_128875_72134 | When did the town WLUJ is licensed in become the capital of the state where The Poor Boob's screenwriter was born? | 1839 |  | 0.000 | 0.000 | 69.917 |
| 2hop__35136_35117 | What's the life span of the scholar who theorized about pulmonary circulation? | 1213–1288 | William Harvey lived from 1578 to 1657. | 0.000 | 0.000 | 10.288 |
| 2hop__685930_110949 | What is the date of birth for the performer of Hits? | 11 September 1962 | October 9 1940 | 0.000 | 0.000 | 40.593 |
| 4hop1__88342_49853_128008_89859 | When does the regular season of the league where the team with the most titles from the series after which the MLB MVP award is given out plays? | March 29, 2018 | April to September | 0.000 | 0.000 | 31.882 |
| 3hop2__304722_692815_63959 | When was the last time Darren Carter's team beat the 1894-95 FA Cup winner? | 1 December 2010 | never | 0.000 | 0.000 | 54.798 |
| 2hop__64537_482364 | Who is the father of the man with the same name as the Big Brother hot dog challenge winner? | Aeson | Matt's father | 0.000 | 0.000 | 28.538 |
| 2hop__29191_80144 | Who became the president of Frederic Chopin's original country in 1990? | Lech Wałęsa | Lech Walesa | 0.000 | 0.500 | 6.438 |
| 3hop1__136129_87694_124169 | What year did the Governor of the city where the basilica named after the same saint as the one that Mantua Cathedral is dedicated to die? | 1952 |  | 0.000 | 0.000 | 69.505 |
| 2hop__46862_68042 | When did the political party that favored harsh punishment of southern states after the Civil War, gain control of the House? | the 2010 election | 1866 | 0.000 | 0.000 | 16.239 |
| 2hop__66167_72467 | Where did the player who scored the most points in a NBA season go in the NBA Draft? | Seattle SuperSonics | Chicago Bulls | 0.000 | 0.000 | 39.632 |
| 2hop__24661_66233 | When does the new season of the Carolina Hurricanes' league start? | October 3, 2018 | October 10 | 0.000 | 0.400 | 14.223 |
| 3hop1__823336_228453_86925 | What is the symbol of the Saints from the headquarters location of Ten High's manufacturer called? | fleur - de-lis | Saints | 0.000 | 0.000 | 28.736 |
| 2hop__488811_30110 | What happened to the Curonians who lived in the area where Erich Zakowski was born? | expelled by the Soviet Union | assimilated into the German population | 0.000 | 0.000 | 34.087 |
| 3hop2__304722_582839_63959 | When was the last time the team employing Ronnie Morris beat the 1894-95 FA Cup winner? | 1 December 2010 |  | 0.000 | 0.000 | 71.771 |
| 2hop__278022_71701 | what year did the Japanese get to city where Wang Yue died and the rest of Guangdong province? | November 5 | 1938 | 0.000 | 0.000 | 20.625 |
| 2hop__88834_356793 | Who is the father of Nim's actress in Return to Nim's Island? | Steve Irwin | Matthew Fanning | 0.000 | 0.000 | 15.342 |
| 3hop1__507961_160713_77246 | What is the meaning of the word that is also a majority religion in India when the country where Banna is located was created in the Arabic dictionary? | the country of India | Hindu | 0.000 | 0.000 | 21.157 |
| 2hop__394720_120259 | When was the kingdom that Heptarchy was part of abolished? | 918 | 1066 | 0.000 | 0.000 | 24.791 |
| 2hop__73069_88165 | Who did the person considered the founder of three major monotheistic religions marry after the death of Sarah? | Keturah | Keturah | 1.000 | 1.000 | 7.207 |
| 3hop1__208108_547811_41132 | How many times did plague occur in the birthplace of La Silvia's composer? | 22 | once | 0.000 | 0.000 | 11.130 |
| 2hop__554167_451128 | Who is the child of the Victim of Romance performer? | Chynna Phillips | Harper Grace Presley | 0.000 | 0.000 | 9.563 |
| 2hop__16632_110014 | What year was the person who described the use of bitumen and pitch to tar ships born? | 1517 | 484 BCE | 0.000 | 0.000 | 10.820 |
| 2hop__130668_126089 | Who was in charge of the state where Graft-De Rijp is located? | Johan Remkes | King Willem-Alexander | 0.000 | 0.000 | 17.979 |
| 2hop__754682_730690 | What movement was Vibullia Alcia Agrippina's child a proponent for? | Second Sophistic |  | 0.000 | 0.000 | 68.924 |
| 3hop2__127483_125104_10557 | What was the language Auctor comes from during the era of Rotrude's father later known as? | Medieval Latin | Latin | 0.000 | 0.667 | 36.852 |
| 2hop__293131_110222 | What is the date of birth of the 'Woman with a Parrot' creator? | 10 June 1819 | October 23 1881 | 0.000 | 0.000 | 11.728 |
| 2hop__77233_33207 | How did did the people fare during the reign of the abolisher of sati partha in India? | a long period of prosperity for the British people | women's welfare improved | 0.000 | 0.000 | 20.427 |
| 2hop__56270_68396 | What percentage of the nation's population lives in the state that includes the 951 area code? | more than 12 percent | 11.8% | 0.000 | 0.000 | 28.565 |
| 2hop__333020_69048 | Who is the Chief Judge of the place where Orezi was born? | Honorable Justice Abiodun Smith | Unknown | 0.000 | 0.000 | 17.923 |
| 4hop3__257340_88460_30152_20999 | How were the same people who the Somali Muslim Ajuran Empire declared independence from expelled from the natural boundary between Thailand and the country where Khong Island is found? | The dynasty regrouped and defeated the Portuguese | Malay people | 0.000 | 0.000 | 41.774 |
| 3hop1__131829_39743_24526 | What is the average winter daytime temperature in the region containing Richmond, in the state where WXBX is located? | upper 40s–lower 50s °F | 48 | 0.000 | 0.000 | 48.397 |
| 3hop1__765565_326948_10972 | When did the birthplace of the Here Come the Girls performer elect its first black mayor? | 1970s | 1973 | 0.000 | 0.000 | 7.688 |
| 2hop__87580_7298 | Along with the person who played Andy's brother in The Office, what notable pop artist started his career in adult contemporary radio? | Michael Bublé |  | 0.000 | 0.000 | 68.624 |
| 2hop__471509_136043 | The Nicole and Natalie album's band is named after who? | Natalie Albino | Nicole and Natalie | 0.000 | 0.400 | 22.276 |
| 4hop1__562740_49925_13759_736921 | Where is the district that the person who wanted to reform and address the religion that practices altarage preach a sermon on Marian devotion before his death located? | Saxony-Anhalt | Umbria | 0.000 | 0.000 | 29.042 |
| 3hop2__76623_90098_10557 | What was the version of the language the last name Sylvester comes from, used in the era of the individual crowned Emperor of the West in 800 CE, later called? | Medieval Latin | Latin | 0.000 | 0.667 | 17.169 |
| 2hop__77134_88165 | Who did the person Sarah married to in the bible marry after the death of Sarah? | Keturah | Keturah | 1.000 | 1.000 | 7.484 |
| 3hop1__145427_106426_157788 | What is one railway line in the country the produced The Longest Night in the city where You Wenhui was born? | Yamanote Line loop | Beijing-Shanghai Railway Line | 0.000 | 0.333 | 37.946 |
| 2hop__122140_37168 | Where does Hala El Badry's university rank according to QS World University Rankings? | 551-600 | 321 | 0.000 | 0.000 | 11.614 |
| 2hop__807257_590611 | Who is a cast member of the show that contains the character Michael Knight? | Deanna Russo | David Hasselhoff | 0.000 | 0.000 | 7.588 |
| 2hop__24976_25002 | how long lasting was the pact between Kravchuk and the person organizers wanted to arrest? | 10-year | one month | 0.000 | 0.000 | 14.292 |
| 3hop2__127483_91678_10557 | What was the language Auctor comes from during the era of the man crowned first Holy Roman Emperor later known as? | Medieval Latin | Latin | 0.000 | 0.667 | 10.202 |
| 4hop1__145494_698949_157828_162309 | When did the country whose co-official language was used in the movie named after the place where Bela Linder died first attend the Olympics games as an independent team? | 2016 | 1920 | 0.000 | 0.000 | 54.838 |
| 4hop1__342858_131850_33952_34109 | What is the PBS station in the second largest city in the state where Yuma's Library District is located? | KUAT-TV 6 | KAZA | 0.000 | 0.000 | 14.033 |
| 3hop1__829944_474097_84283 | Who is the owner of the record label that the performer of Better Than Me is signed with? | Warner Music Group | Warner Music Group | 1.000 | 1.000 | 30.956 |
| 2hop__71336_604644 | Which record label does the band who sang hey now you're an allstar belong to? | 429 Records | Warner Bros. Records | 0.000 | 0.400 | 26.775 |
| 2hop__6827_75783 | When was countrywide bought by the company that bought FleetBoston Financial? | July 1, 2008 | September 2008 | 0.000 | 0.400 | 7.612 |
| 2hop__726892_131968 | Which is the body of water by Leo Bennett's place of death? | River Thames | Mississippi River | 0.000 | 0.500 | 27.016 |
| 4hop1__105688_17130_70784_79935 | When was the region that lies to the north of where Eyal Golan is located and the Persian Gulf created? | 1930 | 1920 | 0.000 | 0.000 | 35.131 |
| 3hop1__222497_309482_89834 | When did the merchant of the city where the composer of Al gran sole carico d'amore takes place? | 16th - century | 1813 | 0.000 | 0.000 | 53.074 |
| 4hop3__857_846_326964_7713 | How long had the city where the Yongle emperor greeted the person to whom the edict was addressed been the capitol of Yaxing Coach's headquarters location? | about 400 years | 22 years | 0.000 | 0.400 | 32.807 |
| 4hop3__710473_354093_254885_76356 | Who pushed for a canal crossing across the continent El Valle is located linking the Pacific Ocean and the ocean Sampit River leads to? | Ferdinand de Lesseps | Theodore Roosevelt | 0.000 | 0.000 | 13.313 |
| 4hop1__151650_5274_458768_33637 | How many ethnic minorities were looked at differently in the city where the headquarters of the only group larger than Desde El Principio's record label is located? | two | New York City | 0.000 | 0.000 | 35.967 |
| 2hop__809785_606637 | What record label does the performer of Adventures in Your Own Backyard belong to? | Secret City Records | Sub Pop Records | 0.000 | 0.333 | 15.268 |
| 3hop1__857_846_7770 | Where does the Yangtze River flow in the city where the Yongle emperor greeted the person to whom the edict was addressed? | past the west side and then north side | Nanjing | 0.000 | 0.000 | 33.758 |
| 2hop__132909_33564 | During which era was the company that makes Nirbhay founded? | From the 1950s to the 1970s | Cold War era | 0.000 | 0.000 | 19.409 |
| 2hop__63483_786853 | Who is the spouse of the person who plays Batman in the Lego Batman Movie? | Amy Poehler | Jennifer Morrison | 0.000 | 0.000 | 39.117 |
| 3hop2__304722_238553_63959 | When was the last time Peter Till's sports team beat the winner of the 1894-95 FA Cup? | 1 December 2010 | never | 0.000 | 0.000 | 27.452 |
| 4hop1__152562_5274_458768_33677 | Vilaiyaadu Mankatha's record label has just one company that is larger than it. When was the most recent Bicycle Friendly Community award given to the city where the headquarters of that larger company lies? | 2013 | 2023 | 0.000 | 0.000 | 16.500 |
| 2hop__107690_124896 | Who was the father of The Portrait of George Dyer Talking's creator? | Nicholas Bacon | Sir Nicholas Bacon | 0.000 | 0.800 | 4.255 |
| 3hop1__302383_503371_21711 | When was the Palau de la Generalitat constructed in the city where Martin from the region where Codo is located died? | built in the 15th century | 1894 | 0.000 | 0.000 | 36.090 |
| 3hop1__109113_720914_27537 | Why did Roncalli leave the city where the painter of Venus with a Mirror died? | for the conclave in Rome | because he was appointed Pope | 0.000 | 0.000 | 64.216 |
| 3hop1__498954_160713_77246 | What is the meaning of the word also known as the majority religion in the area of British India that became India when the country where Mandeer is located was created in arabic dictionary? | the country of India | Hindu | 0.000 | 0.000 | 26.240 |
| 4hop3__324979_836463_161616_77103 | When did the birthplace of Ah Via Musicom's performer become the capital of the state Knowles is from? | 1839 | 1845 | 0.000 | 0.000 | 40.364 |
| 3hop1__321720_650651_7262 | Who was a prominent figure at the radio division of the network that created the version of The Biggest Loser set in the country where Seria is? | Walter Sabo | Didi Mahardika | 0.000 | 0.000 | 11.511 |
| 2hop__85455_158105 | Which two features were played up the person who had the biggest net worth in 2017? | ease of use and enhanced support for Plug and Play |  | 0.000 | 0.000 | 69.546 |
| 2hop__780238_110949 | When was the composer of Till dom ensamma born? | 11 September 1962 | April 15 1740 | 0.000 | 0.000 | 17.132 |
| 3hop1__86965_488922_4093 | What is the term for the prime minister in the country where legislators are elected for six year terms? | Wazir-e-Azam | Prime Minister | 0.000 | 0.000 | 40.312 |
| 4hop1__88342_49853_128008_87812 | When is the opening day of the league of the team with the most titles from the event after which they give out the MLB MVP award? | Thursday, March 29 | April 1 | 0.000 | 0.000 | 55.339 |
| 3hop1__107540_720914_27537 | Why did Roncalli leave the place of death of the maker of The Gozzi Altarpiece? | for the conclave in Rome | exile | 0.000 | 0.000 | 63.537 |
| 3hop1__133264_40769_64047 | When did the rx 350 model of the luxury division of the company that manufactures Scion Fuse change body style? | Sales began worldwide in April 2012 | 2013 | 0.000 | 0.000 | 8.037 |
| 3hop1__751782_2053_5289 | What UK label was bought by the broadcast company that, along with ABC and the original network of Undercovers, is one of the major broadcasters based in New York? | Oriole Records. | EMI | 0.000 | 0.000 | 26.496 |
| 4hop1__94201_642284_131926_89261 | Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico? | the Mississippi River Delta | Mississippi River | 0.000 | 0.800 | 36.977 |
| 2hop__785711_73244 | Where is the origin of the river Augrabies Falls is part of? | Thaba Putsoa | Orange River | 0.000 | 0.000 | 24.788 |
| 3hop2__132957_593597_40768 | When did the maker of Acura Legend, the manufacturer of Scion Fuse and Nissan open US assembly plants? | 1981 |  | 0.000 | 0.000 | 69.119 |
| 2hop__10017_18974 | Who led the military expedition in the province that borders Zhejiang to the south? | Chen Zheng | Lin Zexu | 0.000 | 0.000 | 23.471 |
| 2hop__668407_683671 | Who is the father of Empress Wang's husband? | Yang Xingmi | Kangxi Emperor | 0.000 | 0.000 | 8.216 |
| 3hop1__516535_834494_34053 | What was the wettest year in the city that shares the same county as the ghost town Allen? | 1905 | 1972 | 0.000 | 0.000 | 44.568 |
| 2hop__10900_21567 | Who introduced a system of musical notation in the 14th century used in the area where the Ottoman slave trade flourished? | John Kukuzelis | The Ottomans | 0.000 | 0.000 | 47.204 |
| 4hop3__131820_29339_508306_70744 | Which university has more national championships, university of the state having Fort Hill or university of the state whose primary was won by Edwards besides the state having WEKL? | University of South Carolina | Clemson University | 0.000 | 0.333 | 50.846 |
| 3hop2__222979_133099_40768 | When did Nissan along with the company that makes the Comstar wheel and the company that makes the Scion open US assembly plants? | 1981 | 2005 | 0.000 | 0.000 | 41.900 |
| 4hop1__56635_49925_13759_736921 | The German priest, who wanted to reform the religious denomination now the largest in the US, preached a sermon on Marian devotion soon before his death in which German state? | Saxony-Anhalt | Bavaria | 0.000 | 0.000 | 69.223 |
| 2hop__267938_92763 | What is the enrollment at Jack Clay's employer? | 16,801 | Not enough information | 0.000 | 0.000 | 17.973 |
| 3hop1__64957_87694_64412 | When did the city that contains the basilica named after the saint who is holding a knife in the last supper become a country? | 11 February 1929 | 1861 | 0.000 | 0.000 | 66.607 |
| 2hop__44815_494136 | Who is the spouse of the actor of Ethan in A Dog's Purpose? | Meg Ryan |  | 0.000 | 0.000 | 68.642 |
| 2hop__70338_36240 | What specific part of the book where the quote "by the skin of your teeth" comes from document reference for Mary? | Genesis 3:15 | Job 19:26 | 0.000 | 0.000 | 22.094 |
| 2hop__632234_122868 | Where did the author of The National Dream go to university? | University of British Columbia | Harvard University | 0.000 | 0.333 | 17.532 |
| 2hop__32441_274635 | In which military branch did the other president besides Taft that Jenkins Orphanage played for serve? | United States Army |  | 0.000 | 0.000 | 69.511 |
| 2hop__14_59409 | Who was Beyonce's husband talking about in the the song Cry? | three different relationships he had in the past | Jay-Z | 0.000 | 0.000 | 9.431 |
| 2hop__499577_617190 | Who is the spouse of the child of Peter Andreas Heiberg? | Johanne Luise Heiberg |  | 0.000 | 0.000 | 68.837 |
| 3hop1__178435_547811_80702 | What is the name of the famous bridge in the birthplace of Gloria in D Major's composer? | Rialto Bridge | Mozart Bridge | 0.000 | 0.500 | 19.733 |
| 4hop3__129456_29339_508306_70744 | Between the state university in the state without North Point Mall and where Edwards won the primary and the university in Fort Hill's town which has the more national championships? | University of South Carolina | University of Texas | 0.000 | 0.571 | 28.507 |
| 4hop3__678814_466199_695123_72134 | When did the town WIZE is licensed in become capitol of the state where Stoney Creek Township is located? | 1839 | 1867 | 0.000 | 0.000 | 31.180 |
| 2hop__655505_110949 | What is the Till dom ensamma performer's birth date? | 11 September 1962 | April 15 1966 | 0.000 | 0.000 | 17.140 |
| 2hop__341176_711757 | What other district is found in the same county as Gmina Stężyca? | Gmina Ryki | Gmina Szczytno | 0.000 | 0.500 | 43.000 |
| 2hop__52283_75184 | Who play's Marty's girlfriend in back to the future? | Claudia Wells | Claudia Wells | 1.000 | 1.000 | 9.879 |
| 2hop__693444_220151 | Who is the spouse of Young Man Luther's author? | Joan Erikson | Dorothy L. Outler | 0.000 | 0.000 | 10.903 |
| 2hop__631139_75487 | Who is the federal leader of the party in which John Lynch-Staunton held a position? | Andrew Scheer | Ed Broadbent | 0.000 | 0.000 | 12.142 |
| 3hop1__103890_10659_67187 | Who was the first elected president of the country where, along with the nation where Ágnes Konkoly is from, many expelled French Jews relocated to? | Gabriel Narutowicz | George Washington | 0.000 | 0.000 | 23.331 |
| 2hop__326404_120065 | When did Charles Frederick II's birthplace fall? | 1738 | 1866 | 0.000 | 0.000 | 22.299 |
| 2hop__36747_18974 | Who led the military expedition in the place where the fight between Ming loyalists and Manchus occurred? | Chen Zheng | Hong Taiji | 0.000 | 0.000 | 40.625 |
| 4hop1__37608_32392_823060_610794 | Which county has the location that shares a border with the city that became the state capital of where Representative Thomas Tucker was from? | Richland County | Franklin County | 0.000 | 0.500 | 59.274 |
| 3hop1__635099_131926_13165 | What treaty ceded territory to the US extending west to the body of water by the city where Write This Down was formed? | Treaty of Paris | Oregon Treaty | 0.000 | 0.400 | 29.741 |
| 2hop__862994_69048 | Who is the chief judge at Gloria Kemasuode's birthplace? | Honorable Justice Abiodun Smith | Unknown | 0.000 | 0.000 | 6.414 |
| 2hop__24441_36630 | What countries found their economic growth in the period young boys were employed in the glass making industries? | United Kingdom and Prussia | United Kingdom | 0.000 | 0.667 | 12.699 |
| 4hop1__836115_49925_13759_736921 | Where is the district that the person who wanted to reform and address Vincentas Sladkevičius' religion preached a sermon on Marian devotion before his death located? | Saxony-Anhalt | Šiauliai | 0.000 | 0.000 | 12.490 |
| 4hop1__51465_53706_795904_580996 | What shares a border with the city where the person who went to the state where most billionaires in America live during the gold rush works? | Rio Linda | Nevada | 0.000 | 0.000 | 47.216 |
| 2hop__538735_3299 | What season was the performer of Addicted a part of? | season five | 2015 | 0.000 | 0.000 | 21.255 |
| 4hop3__405277_88460_30152_20999 | How were the people from whom the new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and Ban Hinlat's country? | The dynasty regrouped and defeated the Portuguese | No | 0.000 | 0.000 | 8.155 |
| 3hop1__617062_127905_54974 | What is the farthest home run ever hit in the league where the team that Jim Wilson plays for? | about 575 feet | Insufficient information. | 0.000 | 0.000 | 3.764 |
| 3hop1__475351_160713_77246 | In the Arabic dictionary, what's the meaning of the word for the majority religion in the area of British India that became India when Dourkhan's country was created? | the country of India | Hindū | 0.000 | 0.000 | 29.705 |
| 2hop__203955_25719 | At the end of which year did the tribes from where the Michelsberg culture is from invade the Roman Empire? | 406 | never | 0.000 | 0.000 | 26.547 |
| 2hop__15416_58040 | What is the highest city in the state where Dell ranks sixth by revenue? | Fort Davis | Alpine | 0.000 | 0.000 | 20.110 |
| 2hop__323534_810083 | Where was the author of Marcinkus born? | Scranton | Croatia | 0.000 | 0.000 | 12.237 |
| 3hop1__377943_650651_7262 | Who was a prominent figure at the radio division of the broadcasting company that created the version of The Biggest Loser: The Final Chance, set in the country containing Labu? | Walter Sabo | Datuk Seri Dr. Mohd. Azlan Mohd. Ali | 0.000 | 0.000 | 17.728 |
| 4hop1__570103_49925_13759_736921 | Where is the district that the person who wanted to reform and address Brian Nobles's religion preached a sermon on Marian devotion before his death located? | Saxony-Anhalt | Lourdes | 0.000 | 0.000 | 18.699 |
| 2hop__153573_44085 | What was the show named after the character featured in the video game Mickey's Safari in Letterland? | The Mickey Mouse Club | Mickey's Safari | 0.000 | 0.000 | 14.064 |
| 2hop__135668_561444 | Who is daughter to the third Rebbe of Chabad Lubavitch, named after his father? | Dovber Schneuri | Chaya Mushka | 0.000 | 0.000 | 15.599 |
| 2hop__706038_14251 | What line by a cast member of Birds of Feather is believed to have started the archie nickname? | "Archibald, certainly not!" | Sally Struthers | 0.000 | 0.000 | 18.078 |
| 3hop2__140194_49541_51068 | Who is the actress who plays the role of the Queen of England in 1890, on the network that aired The Dinosaurs!? | Jenna - Louise Coleman | Janet McTeer | 0.000 | 0.000 | 6.274 |
| 2hop__709357_136043 | The Move Ya Body song's band is named after who? | Natalie Albino | James Pointer | 0.000 | 0.000 | 12.659 |
| 2hop__827940_277334 | Which child of the author of One More Time helped adapt it into a play? | Carrie Hamilton | John Penick | 0.000 | 0.000 | 5.302 |
| 4hop1__707078_765799_282674_759393 | What is the capital of the county that shares a border with the county whose capital is the birthplace of Erik Jensen? | Green Bay | Copenhagen | 0.000 | 0.000 | 13.790 |
| 2hop__126306_396277 | What is another notable work by the illustrator of Sylvester and the Magic Pebble? | Doctor De Soto | The Happy Lion | 0.000 | 0.000 | 16.510 |
| 3hop1__518381_709625_84283 | Who owns the record label of the Lovin' All Night performer? | Warner Music Group | Universal Music Group | 0.000 | 0.667 | 12.359 |
| 2hop__28203_55840 | Who played king whose regal Name did Albert Adopt in the king's speech? | Colin Firth | Colin Firth | 1.000 | 1.000 | 5.320 |
| 2hop__30741_85990 | Who were the leaders of the opposition of the party that dominated until the mid-20th century? | anti-slavery activists, modernizers, ex Whigs and ex Free Soilers | Clement Attlee | 0.000 | 0.000 | 14.137 |
| 2hop__6750_6733 | To what division of languages does one with an innovative form belong? | Eastern block | language family | 0.000 | 0.000 | 27.904 |
| 2hop__30905_213732 | Who is the child of the person who sponsored the resolution to reopen trade with Vietnam, along with Kerry? | Meghan McCain | Meghan McCain | 1.000 | 1.000 | 13.962 |
| 2hop__69_59409 | Who was the artist who is associated with Beyonce's premiere solo recording talking about in the song "Cry"? | three different relationships he had in the past | Jay-Z | 0.000 | 0.000 | 27.102 |
| 2hop__96177_121319 | Where did the star of Dark Hazard study? | American Academy of Dramatic Arts | University of Edinburgh | 0.000 | 0.250 | 16.118 |
| 2hop__34594_160249 | Between which years was the war with the person who suffered a great amount of losses battling the Roman army? | 323–272 BC | 58–50 BC | 0.000 | 0.500 | 9.184 |
| 2hop__31085_23241 | According to Pew, in 2010 what percent of Nigeria's population was the same religion as Ibn Tufail? | 48.8 percent | 49.5% | 0.000 | 0.000 | 10.293 |
| 3hop1__857241_503371_21711 | When was the Palau de la Generalitat constructed in the city where Martin from the region where Erla is located died? | built in the 15th century | 1894 | 0.000 | 0.000 | 28.442 |
| 3hop2__326964_263934_7713 | How long had the place of birth of Tang Jinhua been the capital city of the Yaxing Coach headquarters location? | about 400 years | 10 years | 0.000 | 0.400 | 30.558 |
| 2hop__701525_342810 | What county shares a border with Cedar Springs? | Bell County | Glasscock County | 0.000 | 0.500 | 13.781 |
| 4hop2__9988_158279_70784_61381 | When was the region immediately north of the region prevailing with the disgrace of the Near East and the final destination for the export crops established? | 1932 | 1923 | 0.000 | 0.000 | 23.686 |
| 2hop__677929_696450 | Who founded the publisher of The Final Testament of the Holy Bible? | Larry Gagosian | Thomas Nelson, Jr. | 0.000 | 0.000 | 20.446 |
| 2hop__458131_174974 | Who is the spouse of the Rabbit Hole's producer? | Keith Urban | Anne-Marie Casey | 0.000 | 0.000 | 6.568 |
| 2hop__39063_593388 | Who is the father of the person who came up with the concept of the "new" police? | Sir Robert Peel, 1st Baronet | Henry Peel | 0.000 | 0.286 | 5.668 |
| 2hop__12039_11957 | What geological event caused the foundations of the land beneath the city where most university graduates stay after acquiring a degree? | erosion of the Rocky Mountains | retreat of the glaciers | 0.000 | 0.286 | 35.936 |
| 3hop2__132957_595371_40768 | When did the maker of the Acura Legend, the manufacturer of the Scion xB, and Nissan, open US assembly plants? | 1981 | 1982, 1988, 1984 | 0.000 | 0.000 | 42.643 |
| 2hop__683835_709167 | What award did the author of The Red Tree receive? | Academy Award for Best Animated Short Film | British Book Design and Production Award | 0.000 | 0.154 | 3.779 |
| 3hop1__617062_127905_80487 | When is the All-Star game being played in the league that Jim Wilson plays for? | July 11, 2017 | Cannot determine without knowing Jim Wilson's league. | 0.000 | 0.000 | 4.120 |
| 3hop1__446612_160545_60577 | Where did they film The Beach in the country where Ercole Manfredi died? | Ko Phi Phi Leh |  | 0.000 | 0.000 | 49.920 |
| 3hop1__702699_792411_51423 | What is the name of the castle in the birth city of the performer on the album Maria? | Casa Loma |  | 0.000 | 0.000 | 49.786 |
| 3hop1__104762_24918_24991 | What went down after the Soviet president visiting the country of citizenship of Yasyn Khamid while the protests were taking place departed from the Kremlin? | Soviet flag | 1991 August Coup | 0.000 | 0.000 | 20.316 |
| 4hop3__839787_88460_30152_20999 | How were the same people who the Somali Muslim Ajuran Empire declared independence from expelled from the natural boundary between Thailand and the country where Sepon mine is found? | The dynasty regrouped and defeated the Portuguese | Khmer | 0.000 | 0.000 | 30.066 |
| 3hop1__160324_131905_41948 | What body of water was dry land during the last ice age, besides the Irish Channel and the sea near the former Atlantic Hotel? | English Channel | English Channel | 1.000 | 1.000 | 22.782 |
| 3hop1__103180_214799_259594 | What is the country of the border troops of the country of the literature of the country Wilfried Grobner was in? | GDR | Austria | 0.000 | 0.000 | 17.142 |
| 3hop1__254856_400692_51423 | What is the name of the castle found in the birthplace of the performer of Falling Out? | Casa Loma | Castle of the Holy Spirit | 0.000 | 0.000 | 47.091 |
| 3hop2__90098_125104_10557 | What was the form of the language that the last name Sylvester comes from, used in the era of Rotrude's father, later known as? | Medieval Latin | Old French | 0.000 | 0.000 | 22.146 |
| 4hop1__88342_49853_128008_54974 | What is the longest home run hit in the league where the team with the most titles from the event after which the MLB MVP is awarded plays? | about 575 feet | Mickey Mantle | 0.000 | 0.000 | 49.163 |
| 4hop3__279362_88460_30152_20999 | How were people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the natural boundary between Thailand and Vat Yotkeo's country? | The dynasty regrouped and defeated the Portuguese | No known connection. | 0.000 | 0.000 | 9.043 |
| 2hop__31124_31122 | What century did a notable subjective idealist live in? | 18th | 18th century | 0.000 | 0.667 | 4.923 |
| 2hop__466700_214359 | Who is one of the members of the music group that performed "Good Little Girls"? | Aaron Benward | Adam Schlesinger | 0.000 | 0.000 | 7.826 |
| 3hop2__326964_7861_7713 | How long had the city containing some of the finest and oldest museums in China been the capitol city of Yaxing Coach's headquarters location? | about 400 years | since 1949 | 0.000 | 0.000 | 25.361 |
| 4hop1__58323_375563_161848_83118 | When does daylight savings end in the country near the country where the author of The Book Thief is from? | the first Sunday in April | first Sunday in October | 0.000 | 0.750 | 35.691 |
| 3hop1__773623_87694_124169 | On what date did Governor of the location of the Basilica named after the same saint as Đakovo Cathedral and the head of the catholic religion end? | 1952 | April 2 2005 | 0.000 | 0.000 | 44.384 |
| 3hop2__326964_846_7713 | How long had the city where the Yongle Emperor greeted the Karmapa been the capitol city of Yaxing Coach's headquarters location? | about 400 years | 226 years | 0.000 | 0.400 | 33.461 |
| 3hop1__389069_132457_47686 | Where do greyhound buses depart from, in the city where the performer of Sleep Is the Enemy was formed? | Toronto Coach Terminal |  | 0.000 | 0.000 | 49.819 |
| 4hop1__131611_32392_823060_610794 | In which county is the city sharing a border with the capitol of the state where Ocean Forest Country Club is located? | Richland County | Lee County | 0.000 | 0.500 | 14.077 |
| 2hop__458672_20057 | Who started the foundation that developed the Free Software Directory? | Richard Stallman | Richard Stallman | 1.000 | 1.000 | 4.579 |
| 2hop__19007_60935 | Who was the first person to do a full translation of the Bible into the script in which Hokkien is sometimes written? | St Jerome | Thomas Barclay | 0.000 | 0.000 | 13.442 |
| 2hop__623501_297043 | What watercourse is the river on which the Lostock Dam is located the mouth of? | Hunter River | Mersey | 0.000 | 0.000 | 15.381 |
| 2hop__121386_37168 | According to QS World University Rankings, where does the college that Ibrahim Shihata attended rank? | 551-600 | 428 | 0.000 | 0.000 | 11.069 |
| 2hop__789565_416401 | Who is the child of Empress Dowager Huang's child? | Wang Jipeng | Emperor Jing of Han | 0.000 | 0.000 | 35.798 |
| 3hop1__136016_87694_124169 | On what date did Governor of the location where the basilica named after the same saint as Leeds Minster and the head of the catholic religion end? | 1952 | 2013 | 0.000 | 0.000 | 34.889 |
| 3hop2__809962_89048_66294 | Who played the man named identically to the singer of Is She Really Going Out With Him in the movie about the composer of Speechless? | Lawrence Hilton - Jacobs |  | 0.000 | 0.000 | 50.258 |
| 4hop2__71753_729371_70784_79935 | When was the region that lies to the north of where Israel is located and the location of Operation Earnest Will established? | 1930 | Golan Heights 1967 | 0.000 | 0.000 | 20.894 |
| 3hop1__742464_15840_36002 | What were the Genesis's advantages over the platform in which Robodemons is played? | built on 16-bit architectures and offered improved graphics and sound | superior graphics and sound | 0.000 | 0.429 | 14.779 |
| 3hop1__220023_503371_21711 | When was the Palau de la Generalitat constructed in the city where Martin from the region where Trasobares is located died? | built in the 15th century | 1894 | 0.000 | 0.000 | 30.486 |
| 4hop1__726152_153080_159767_81096 | Who won the Indy car race in the largest city and capital of the state where the performer of Tijuana Moods was born? | Mario Andretti |  | 0.000 | 0.000 | 49.735 |
| 2hop__814408_155922 | Into what does the river for which Toongabbie Creek serves as the mouth flow? | Sydney Harbour | Hawkesbury River | 0.000 | 0.000 | 10.074 |
| 4hop1__726152_153080_33897_81096 | Who won the Indy Car Race in the largest populated city of the state where the performer of Tijuana Moods is from? | Mario Andretti |  | 0.000 | 0.000 | 47.628 |
| 2hop__17508_17335 | When did the person all of the diadochi painted themselves to look like die? | 323 BC | 305 BCE | 0.000 | 0.000 | 19.214 |
| 2hop__82766_67668 | The actor playing Barney in How I Met Your Mother played who in Batman Under The Red Hood? | Nightwing / Dick Grayson | Joker | 0.000 | 0.000 | 12.509 |
| 2hop__151455_120171 | What was the last year of the Castle of Riddles publisher? | 1986 | 1983 | 0.000 | 0.000 | 9.224 |
| 3hop1__109113_720914_41132 | How many times did plague occur in the place where Venus with a Mirror's creator died? | 22 | one | 0.000 | 0.000 | 25.975 |
| 3hop2__57233_105140_56883 | During the war in which The Things They Carried is set, when was conscription introduced by the country where the film Grievous Bodily Harm was later released? | 1964 | 1916 | 0.000 | 0.000 | 25.751 |
| 3hop1__346812_15840_36002 | What were the Genesis's advantages over the platform that plays Drac's Night Out? | built on 16-bit architectures and offered improved graphics and sound | Genesis had better graphics and sound compared to the platform that played Drac's Night Out. | 0.000 | 0.250 | 24.609 |
| 2hop__827755_156034 | What is the river that the Kettle Generating Station is located on a tributary of? | Hudson's Bay | Columbia River | 0.000 | 0.000 | 5.361 |
| 2hop__507722_124896 | Who is the father of the author of Novum Organum? | Nicholas Bacon | Sir Nicholas Bacon | 0.000 | 0.800 | 3.127 |
| 2hop__96137_121319 | Where did the actor in The Woman in the Window study or work? | American Academy of Dramatic Arts | University of Kansas and theater | 0.000 | 0.200 | 5.723 |
| 3hop1__120881_2053_52946 | When is Celebrity Big Brother coming to the network which, along with ABC and the network which released Stand Against Fear, is the other major broadcaster based in NY? | February 7, 2018 | 2001 | 0.000 | 0.000 | 30.838 |
| 2hop__391873_88622 | Who is the head of the agency that created MYSTIC? | ADM Michael S. Rogers | Paul Nakasone | 0.000 | 0.000 | 19.851 |
| 4hop1__88342_49853_128008_86588 | What's the longest homer in the history of the league where the team with the most titles from the event after which they give out the MLB MVP award plays? | 582 feet (177 m) | The Shot | 0.000 | 0.000 | 22.908 |
| 2hop__511454_120259 | When was Lady Godiva's birthplace abolished? | 918 | never | 0.000 | 0.000 | 11.930 |
| 3hop1__516535_834494_34088 | Where are NASCAR races held in the city in the same county as Allen? | Tucson Raceway Park | none | 0.000 | 0.000 | 18.190 |
| 2hop__84103_345851 | The actor who plays the girl in Spiderman Homecoming is signed to which record label? | Hollywood Records | Def Jam Recordings | 0.000 | 0.000 | 14.776 |
| 4hop1__443274_17130_70784_79935 | When was the region immediately north of the region where the country in which Sar-El can be found is located and the Persian Gulf created? | 1930 | 1932 | 0.000 | 0.000 | 29.192 |
| 3hop1__857_846_7713 | How long was the city where the Yongle Emperor greeted the leader that the edict was addressed to, the capital city of Yangzhou? | about 400 years | Beijing | 0.000 | 0.000 | 39.070 |
| 4hop1__749363_49925_13759_736921 | In what state is the district where the man who wanted to reform the religion practiced by Innocenzo Ferrieri preached a sermon on Marian devotion? | Saxony-Anhalt | Maryland | 0.000 | 0.000 | 18.518 |
| 2hop__153009_44880 | Who is the mayor presiding now where Merrill Elam was born? | Lance Bottoms | Eric Johnson | 0.000 | 0.000 | 4.126 |
| 2hop__42578_55840 | Who played the monarch at the end of WW2 in The King's Speech? | Colin Firth | Colin Firth | 1.000 | 1.000 | 6.323 |
| 2hop__542569_87295 | Who does the performer of A Spoonful of Sugar play in Princess Diaries? | Queen dowager Clarisse Renaldi | no role | 0.000 | 0.000 | 18.781 |
| 2hop__50910_177869 | Who married the publisher of abolitionist newspaper The North Star? | Helen Pitts Douglass | Anna Murray Douglass | 0.000 | 0.333 | 4.644 |
| 2hop__61924_712629 | Who is the spouse of the current queen of England? | Philip Mountbatten | Camilla | 0.000 | 0.000 | 10.851 |
| 2hop__283_574195 | Where was the person who said she is the reigning national voice educated? | National Institute of Dramatic Art | University of London | 0.000 | 0.250 | 23.483 |
| 2hop__702496_430061 | What county was Tim Dubois born in? | McDonald County | St. Bernard Parish | 0.000 | 0.000 | 6.306 |
| 2hop__60229_38663 | What was the percentage the Dow Jones fell during the time the first cake mix come out? | 54.7% | 0% | 0.000 | 0.000 | 24.194 |
| 4hop1__310029_153080_159767_81096 | Who won the Indy Car Race in the largest city and capital of the state where the performer of Jazz Portraits: Mingus in Wonderland is from? | Mario Andretti | Scott Dixon | 0.000 | 0.000 | 31.269 |
| 2hop__511296_2684 | What was the name of the law passed by the actor from Terminator? | "Donda West Law" | AB 32 | 0.000 | 0.000 | 6.171 |
| 2hop__84847_21969 | When was the 1st governor general of India impeached? | 1786 | 1795 | 0.000 | 0.000 | 17.080 |
| 2hop__557743_92763 | What is the enrollment at Henry Latimer's alma mater? | 16,801 | enrollment unknown | 0.000 | 0.000 | 5.514 |
| 2hop__57209_67668 | The actor who plays Barney in "How I Met Your Mother" plays what character in Batman Under the Red Hood? | Nightwing / Dick Grayson | Neil Patrick Harris plays the Joker in "Batman Under the Red Hood." | 0.000 | 0.000 | 40.040 |
| 2hop__846844_47134 | What is the mascot of the school that owns Goss Stadium at Coleman Field? | Benny Beaver | Spartans | 0.000 | 0.000 | 4.867 |
| 2hop__814856_110882 | What is the Smile performer's birthday? | 1 January 1986 | June 20 1945 | 0.000 | 0.000 | 22.121 |
| 3hop2__607269_223623_162182 | In what region of the country containing Hạ Hòa is John Phan's birthplace found? | South Central Coast | Phú Thọ | 0.000 | 0.000 | 4.690 |
| 3hop1__390673_228453_10972 | When did the headquarters city of Kentucky Tavern's manufacturer elect its first black mayor? | 1970s | 1985 | 0.000 | 0.000 | 13.745 |
| 3hop1__145427_106426_77199 | When did the USA declare war on the country that produced The Longest Night in the city where You Wenhui was born? | December 8, 1941 | December 11 1941 | 0.000 | 0.667 | 5.191 |
| 2hop__68981_39234 | What empire brought the practices of the most common religion in Southeast Asia? | Hephthalite Empire, | Khmer Empire | 0.000 | 0.500 | 15.313 |
| 2hop__29893_29898 | Which affiliate left the company that owns and operates most of the CBC television stations due to an agreement with CHBC? | CFJC-TV in Kamloops | CTV | 0.000 | 0.000 | 11.915 |
| 4hop1__88342_49853_128008_80286 | Who has the most hits in the league whose member team has the most wins of the championship series which proceeds the MVP award? | Pete Rose | Ty Cobb | 0.000 | 0.000 | 23.247 |
| 3hop1__857_846_7810 | What was where the Yongle emperor greeted who the edict addressed GDP per person in 2013? | RMB 98,174(US$16041) | no | 0.000 | 0.000 | 5.843 |
| 3hop1__765388_42197_18397 | Where did the arguer that Vadim Spiridonov's country of citizenship had become an imperialist power declare he would intervene in the Korean conflict? | the Politburo | United Nations | 0.000 | 0.000 | 16.252 |
| 2hop__88149_807969 | The all time top goal scorer in premier league in one season is a member of what team? | Egypt national football team | Blackburn Rovers | 0.000 | 0.000 | 3.509 |
| 2hop__107238_64918 | Who was president when congress declared war on the country where Glumpers was filmed? | President McKinley | Franklin D. Roosevelt | 0.000 | 0.000 | 7.544 |
| 4hop1__3496_5274_458768_33637 | How many ethnic minorities were looked at differently where the only group larger than BMG's partner from 2005 to 2007 is headquartered? | two | one | 0.000 | 0.000 | 20.678 |
| 2hop__538412_126089 | Who was in charge of the province where Schermer is located? | Johan Remkes | Liesbeth Huijbregts | 0.000 | 0.000 | 5.851 |
| 4hop1__629520_49925_13759_736921 | In what region of Germany is the district where a sermon on Marian devotion was preached by the man who wanted to reform the religion of Victor Razafimahatratra? | Saxony-Anhalt | Bavaria | 0.000 | 0.000 | 13.730 |
| 2hop__90773_715759 | What was the inventor of the atomic bomb trigger a participant in? | Manhattan Project | Manhattan Project | 1.000 | 1.000 | 2.788 |
| 3hop1__131783_131926_87157 | What is the direction of flow of the body of water by the city where The Minikahda Club is located? | rises in northern Minnesota and meanders slowly southwards | south | 0.000 | 0.000 | 5.051 |
| 2hop__559076_55984 | Who wrote the song The Story sung by the performer of Give Up the Ghost? | Phil Hanseroth | Robbie Robertson | 0.000 | 0.000 | 5.737 |
| 2hop__286268_97805 | Which war did the operator of the R class patrol boat serve in? | Winter War | World War II | 0.000 | 0.400 | 6.764 |
| 3hop1__462960_160545_62931 | Where was the film The Beach filmed in the country where Pao Sarasin was born? | island Koh Phi Phi | Phang Nga | 0.000 | 0.000 | 21.426 |
| 2hop__159146_120065 | What year did the region where Leuna was located come to an end? | 1738 | 1918 | 0.000 | 0.000 | 25.914 |
| 4hop1__131632_32392_823060_610794 | What county contains the city with a radio station that broadcasts to the capital city of the state where the Peace center is located? | Richland County | Oklahoma County | 0.000 | 0.500 | 29.593 |
| 2hop__20536_96912 | Who gives out the award named after the person who argued that colonies were redundant? | University of Cambridge | British Academy | 0.000 | 0.000 | 15.798 |
| 2hop__837900_71701 | What date did the Japanese get to the city where New Plaza Stadium is located, and the rest of the Guangdong province? | November 5 | October 1938 | 0.000 | 0.000 | 7.918 |
| 3hop1__140554_2053_52946 | When is Celebrity Big Brother coming to the network which, along with ABC and the network which hosts Westinghouse Playhouse, is the other major broadcaster based in NY? | February 7, 2018 |  | 0.000 | 0.000 | 40.467 |
| 3hop2__326964_7719_7713 | How long had the place where the Jin court reestablished itself after fleeing been the capitol city of the headquarters location of Yaxing Coach? | about 400 years | 127 years | 0.000 | 0.400 | 16.232 |
| 2hop__129499_85379 | When did the region where Poachie Range is located achieve U.S. statehood? | February 14, 1912 | 1864 | 0.000 | 0.000 | 13.883 |
| 4hop1__88342_75218_128008_92225 | When is the opening day of the league, that includes the team that played in the most games. of the event that the MLB MVP award is handed out after? | March 29, 2018 | April 1 | 0.000 | 0.000 | 14.930 |
| 2hop__711946_269414 | What record label is the performer who released All Your Faded Things on? | Kill Rock Stars | RCA Records | 0.000 | 0.000 | 2.423 |
| 3hop2__304722_466316_63959 | When was the last time the team that Arthur James was a member of beat the 1894-95 FA cup winner? | 1 December 2010 | 1901 | 0.000 | 0.000 | 23.663 |
| 4hop1__33012_53706_795904_580996 | What city shares a border with the place where the person who went to the state known for its Mediterranean climate during the gold rush worked? | Rio Linda | Las Vegas | 0.000 | 0.000 | 15.869 |
| 3hop1__617062_127905_87812 | When is the opening day of the league that Jim Wilson plays for? | Thursday, March 29 | Unknown | 0.000 | 0.000 | 9.315 |
| 4hop1__107309_457883_650651_7262 | The Embassy of the country where palitaw comes from is found in Bandar Seri Begawan in Country A. Who was a prominent figure at the radio division of the creator of The Biggest Loser at Country A? | Walter Sabo | Mark Cuban | 0.000 | 0.000 | 18.497 |
| 2hop__3739_13529 | When was the football star who backed out due to relay controversy signed by Barcelona? | June 1982 | 2009 | 0.000 | 0.000 | 28.310 |
| 2hop__418941_429437 | What is the mouth of the river which serves as the mouth of the Bumping River? | Yakima River | the mouth of the Bumping River | 0.000 | 0.333 | 13.769 |
| 4hop1__344155_765799_282674_759393 | What is the capital of the county that shares a border with the county where KRSU is licensed to broadcast? | Green Bay | Hutchinson | 0.000 | 0.000 | 33.043 |
| 3hop1__103751_24918_24991 | What went down after the Soviet President visiting the country of origin of Ethella Chupryk while the protests were taking place departed from the Kremlin? | Soviet flag | Baltic independence movement | 0.000 | 0.000 | 18.786 |
| 2hop__40270_11402 | According to the agency that considers if Los Angeles County is to be a separate metropolitan area, what is the total area in square miles? | 17.037 square miles | 4083 | 0.000 | 0.000 | 28.337 |
| 2hop__763840_144857 | What city was the author of Neighbors born in? | Clatskanie | Raleigh | 0.000 | 0.000 | 5.746 |
| 2hop__141365_417697 | What is Mohamed Atta's car, manufactured by the company that includes Datsun, an instance of? | Nissan Altima | Nissan Sentra | 0.000 | 0.500 | 7.843 |
| 4hop3__130276_29339_508306_70744 | Who won more national championships between the university featuring Fort Hill and the university of the state where Edwards won the primary besides the state containing Redan High School? | University of South Carolina | University of Georgia | 0.000 | 0.571 | 35.196 |
| 3hop1__640171_228453_10972 | When did the headquarters city of the Blanton's manufacturer elect its first black mayor? | 1970s | 1984 | 0.000 | 0.000 | 4.620 |
| 2hop__589264_150107 | Who published the Communications of the group that Zhengyou Zhang is a member of? | Association for Computing Machinery | Microsoft Research | 0.000 | 0.000 | 27.436 |
| 3hop1__79039_131926_89261 | Where does the body of water by the home of the Juicy Lucy empty into the gulf of Mexico? | the Mississippi River Delta | Gulf of Mexico | 0.000 | 0.000 | 21.142 |
| 3hop1__358853_339990_54675 | What mountain can you see from Portland, in the state that Raven Creek is located in? | Tualatin Mountains | Mount Tabor | 0.000 | 0.000 | 9.612 |
| 2hop__37097_86208 | When does the river that the majority of Egyptians live near flood each year? | August | July and August | 0.000 | 0.500 | 7.393 |
| 3hop1__639955_834494_34088 | Where are NASCAR races held in the city that shares a county with Helvetia? | Tucson Raceway Park | no NASCAR races | 0.000 | 0.000 | 31.032 |
| 3hop1__726561_132457_47686 | Where do greyhound buses leave from in the city the performer of Born a Lion formed in? | Toronto Coach Terminal | Manchester Piccadilly Station | 0.000 | 0.000 | 24.118 |
| 2hop__424908_500483 | What shares a border with the administrative district where Bieliny, Świętokrzyskie Voivodeship is the capital? | Gmina Daleszyce | Łuków County | 0.000 | 0.000 | 20.452 |
| 4hop2__161602_474028_88460_20999 | How were the Portuguese expelled from the natural boundary between the country that hosted the tournament and the country where A Don is located? | The dynasty regrouped and defeated the Portuguese | by the British in 1808 | 0.000 | 0.000 | 28.323 |
| 2hop__159741_161674 | What month did the person who retained trust in Longchamp go away to the Holy Land? | October | May | 0.000 | 0.000 | 16.494 |
| 3hop2__90098_60649_10557 | What was the language from which the last name Sylvester originates during the era of the Frankish king who formed the Holy Roman Empire later known as? | Medieval Latin | Latin | 0.000 | 0.667 | 7.297 |
| 2hop__809096_491515 | Where did William Beckford's child die? | Bath | Isle of Wight | 0.000 | 0.000 | 9.527 |
| 2hop__24697_634490 | Who is the sibling of Shu Yu from the Chinese capital that Kanmu modeled his government after? | King Cheng of Zhou | Yi Yu | 0.000 | 0.000 | 15.196 |
| 4hop1__814776_765799_282674_759393 | What is the seat of the county sharing a border with the county where Western Islands is headquartered? | Green Bay | Austin | 0.000 | 0.000 | 20.656 |
| 2hop__47712_25719 | At the end of which year did tribes from the place where the Celts are said to have migrated from invade the Roman empire? | 406 | 105 BCE | 0.000 | 0.000 | 8.547 |
| 4hop1__40316_497223_15840_36002 | What were the Genesis's advantages over the game system with a 3 letter abbreviation, featuring a game named after the football league that the Los Angeles Rams are representative of? | built on 16-bit architectures and offered improved graphics and sound | better graphics and sound | 0.000 | 0.429 | 18.955 |
| 2hop__821197_368148 | In which county is Southern Maryland Electric Cooperative headquartered? | Charles County | Charles County | 1.000 | 1.000 | 5.239 |
| 3hop1__699454_831637_91775 | What does seal stand for in the U.S. military branch that operates a list of destroyer classes, as well as the USS Sable? | Sea, Air, and Land | Navy | 0.000 | 0.000 | 24.889 |
| 2hop__71715_50429 | What is the population of the state where Dukes of Hazzard took place? | 10,310,371 (2016 est.) | North Carolina 10.5 million | 0.000 | 0.000 | 27.509 |
| 3hop2__11282_57233_56883 | The native country of emu birds started conscription. This occurred during the war depicted in The Things They Carried. What year did the draft start? | 1964 | 1965 | 0.000 | 0.000 | 3.928 |
| 2hop__166824_185403 | What league does the team that plays in Stadio Ciro Vigorito play for? | Lega Pro Prima Divisione | Serie B | 0.000 | 0.000 | 12.805 |
| 3hop1__354480_834494_34088 | Where do they hold NASCAR races in the city that is in the same county as the Eric Marcus Municipal Airport? | Tucson Raceway Park | Daytona International Speedway | 0.000 | 0.000 | 16.842 |
| 4hop1__638988_17130_70784_79935 | When was the region that lies to the north of the region where Aluf is located and the Persian Gulf created? | 1930 | 1920 | 0.000 | 0.000 | 18.983 |
| 2hop__857585_14251 | What line by the Southern Roses cast member is believed to have started the Archie nickname? | "Archibald, certainly not!" | I'm the king of the hill | 0.000 | 0.000 | 10.613 |
| 3hop2__326964_7855_7713 | How long had the city considered to be a place that is popular with tourists been the capital city of the location of the Yaxing Coach headquarters? | about 400 years | 69 years | 0.000 | 0.400 | 26.287 |
| 2hop__178922_5385 | How many people work in the publisher of World Literature Today? | 11,900 | 15 | 0.000 | 0.000 | 6.424 |
| 2hop__622637_120171 | What year did the company that published Starship Command end? | 1986 | 1986 | 1.000 | 1.000 | 6.482 |
| 2hop__685280_510757 | The Unwinding author volunteered for which organisation? | Peace Corps | Crisis Center | 0.000 | 0.000 | 13.181 |
| 2hop__130867_54221 | What time of day can you start purchasing alcohol in the state that includes Narraguagus High School? | 5am | 7 AM | 0.000 | 0.000 | 5.735 |
| 3hop1__617062_127905_89859 | When does the sport which has Jim Wilson's Cleveland team as a member start the regular season? | March 29, 2018 | September | 0.000 | 0.000 | 11.923 |
| 3hop1__465684_160545_60577 | Where did they film The Beach in the country where Bodindecha died? | Ko Phi Phi Leh | Thailand | 0.000 | 0.000 | 12.089 |
| 2hop__96410_159054 | What was the story that included the character from The Bourne Betrayal based on? | the novel of the same name by Robert Ludlum | David Webb | 0.000 | 0.000 | 29.929 |
| 4hop3__630723_88460_30152_20999 | How were people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the natural boundary between Thailand and Phommathat's country of citizenship? | The dynasty regrouped and defeated the Portuguese | The question contains incorrect geographical and historical information. | 0.000 | 0.167 | 10.055 |
| 2hop__695364_513121 | Who is the child of the male star of the 1921 film Money? | Anthony Ainley | Valentina Valentino | 0.000 | 0.000 | 21.717 |
| 2hop__693157_131890 | What is the body of water by the area where Athlete was formed? | River Thames | River Clyde | 0.000 | 0.500 | 3.610 |
| 3hop2__104979_23308_28343 | When did the troops of the country blamed for the stagnation and failure of the Tripartite discussion leave the country known for Afghan bread? | February 1989 | August 2021 | 0.000 | 0.000 | 15.324 |
| 3hop1__617062_127905_46748 | What is the longest win streak in the league that Jim Wilson's team plays for? | 26 | 17 | 0.000 | 0.000 | 27.778 |
| 4hop1__370129_621192_10659_59955 | What country is between Lithuania and the country many expelled French Jews relocated to, along with the country of citizenship of Marie Antoinette's mother? | Kaliningrad Oblast | Germany | 0.000 | 0.000 | 24.554 |
| 2hop__16446_58556 | What is the name of the food safety system of the federal agency that regulates prescription drugs? | Food Safety Modernization Act (FSMA) | Food Safety Modernization Act | 0.000 | 0.889 | 7.731 |
| 4hop1__726675_508773_85832_745702 | Who is the child of the Italian navigator who explored the eastern coast of the continent César Gaytan was born in for the English? | Sebastian Cabot | Diego Columbus | 0.000 | 0.000 | 27.166 |
| 4hop1__39871_314549_131976_56103 | When do hurricanes occur in the ocean that borders the state where Williamsburg, Main Street Station and another share a track? | usually in the summer or fall | June to November | 0.000 | 0.000 | 5.890 |
| 2hop__518354_67465 | When did the performer of Blue Slide Park release Best Day Ever? | March 11, 2011 | 2003 | 0.000 | 0.000 | 13.348 |
| 3hop2__14777_8533_9331 | What dissolved privileges in Poland held by the empire that used Pan-Slavism as a tool, the leader of the North German Confederation during the Franco-Prussian War, and the Habsburg Monarchy? | March Constitution of Poland | Russian Empire, Bismarck, Habsburg Monarchy | 0.000 | 0.000 | 40.087 |
| 3hop1__79039_131926_90707 | Where does the body of water by the city home to the Juicy Lucy and the Ohio River meet? | at the city of Cairo, Illinois | Cairo, Illinois | 0.000 | 0.571 | 13.051 |
| 3hop2__222979_132477_40768 | When did the maker of the Acura RLX, the owner of Scion, and Nissan open US assembly plants? | 1981 | 1982, 1988, 1984 | 0.000 | 0.000 | 24.988 |
| 2hop__71900_712629 | Who is the spouse of the longest reigning British monarch? | Philip Mountbatten | Prince Philip, Duke of Edinburgh | 0.000 | 0.286 | 5.204 |
| 2hop__506035_325512 | In which county was Larkin I. Smith born? | Pearl River County | Steuben County | 0.000 | 0.400 | 5.396 |
| 4hop3__152056_698586_89595_54362 | Who burned down the city where Dunn Dunn's recording artist died during the conflict after which the historical time period of A Rose for Emily occurred? | Confederate Gen. John Bell Hood | Union Army | 0.000 | 0.000 | 21.539 |
| 4hop1__161765_53706_795904_580996 | What city shares a border with the city where the person who went to the state where Englebart died during the gold rush works? | Rio Linda | Stockton | 0.000 | 0.000 | 25.353 |
| 2hop__656411_345851 | The singer, who starred in Shake It Up, signed to which record label? | Hollywood Records | Epic Records | 0.000 | 0.500 | 6.743 |
| 3hop1__593597_40769_64047 | When did the luxury division of the manufacturer of Scion Fuse change the body style of the RX 350? | Sales began worldwide in April 2012 | 2014 | 0.000 | 0.000 | 10.398 |

</details>
