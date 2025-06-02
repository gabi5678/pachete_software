import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


from functii import detect_outliers_iqr, plot_boxplots,assign_region

data = pd.read_excel('dataIN/fashion.xlsx',sheet_name='Sheet1',index_col=0)
data["Annual Income (k$)"] = data["Price"] * 10 + 30
data["Spending Score"] = data["Sales Volume"] + np.random.randint(-10, 10, size=len(data))
data["Genre"] = np.random.choice(["Male", "Female"], size=len(data))
data["Age"] = np.random.randint(18, 65, size=len(data))

st.markdown("""
<style>
.title{
    color:#fa82f0 !important;
}
.custom-title {
    color: #b5029d !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Proiect Analizare Vânzări Fashion</h1>',unsafe_allow_html=True)
st.markdown('<h3 class="custom-title">Antonescu Gabriela & Guguloi Iulia</h3>', unsafe_allow_html=True)

# Meniu de navigare lateral
section = st.sidebar.radio("Navigați la:",
                           ["Descriere dataset",
                            "Informatii privind DataFrame-ul",
                            "Valori lipsa ale datasetului",
                            "Tratare valori lipsa",
                            "Analiza valorilor extreme",
                            "Codificare și scalare",
                            "Identificare si stergere duplicate",
                            "Grupare dupa Categorie",
                            "Predictie pret",
                            "Analiză Geografică",
                            "Clusterizare clienți",
                            "Clasificare logistică",
                            "Regresie multiplă",
                            ], key="radio_navigare")

if section == "Descriere dataset":
    st.header("📊 Analiza datasetului Fashion Sales")
    st.markdown("""
    ### 🛍️ Despre Fashion Sales Dataset
    **Fashion Sales Dataset** este un set de date detaliat, conceput pentru a simula procesul real de vânzări în industria modei. Acesta oferă o perspectivă valoroasă asupra pieței modei și reprezintă o resursă esențială pentru analiza tendințelor de vânzare, prognoza cererii și optimizarea strategiilor de afaceri.

    📌 **Caracteristici principale:**
    - 🛒 **Proces de vânzări realist** – Include detalii despre achizițiile clienților, tranzacții și caracteristicile produselor.
    - 📆 **Actualizări săptămânale** – Reflectă cele mai recente tendințe și preferințe ale consumatorilor.
    - 🏷️ **Diverse atribute** – Cuprinde informații despre prețuri, branduri, categorii, rating-uri, review-uri, dimensiuni disponibile, culori și istoric de achiziții.
    - 🧑‍🤝‍🧑 **Interacțiuni cu clienții** – Conține date despre reviste de modă, influenceri și impactul rețelelor sociale asupra deciziilor de cumpărare.
    - 🍂 **Variații sezoniere** – Permite analiza preferințelor de modă în funcție de anotimpuri și perioade specifice din zi.
    - 📊 **Eșantion semnificativ** – Numărul mare de observații asigură un set de date robust pentru analize detaliate și prognoze precise.
    """)

if section == "Informatii privind DataFrame-ul":
    st.header('Dataset')
    st.dataframe(data.head(1000))

if section == "Valori lipsa ale datasetului":
    if data.isnull().sum().sum() == 0:
        st.success("✅ Nu există valori lipsă în dataset!")
    else:
        st.warning("⚠️ Există valori lipsă!")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

if section == "Tratare valori lipsa":
    st.header("🧼 Tratarea valorilor lipsă")

    missing = data.isnull().sum()
    missing_percent = (missing / len(data)) * 100
    missing_df = pd.DataFrame({"Valori Lipsă": missing, "Procent": missing_percent})
    missing_df = missing_df[missing_df["Valori Lipsă"] > 0]

    if missing_df.empty:
        st.success("✅ Nu există valori lipsă de tratat!")
    else:
        st.dataframe(missing_df)

        metoda_selectata = st.radio(
            "Alege metoda de tratare:",
            ("Completare cu mediana/moda", "Eliminare rânduri cu valori lipsă", "Completare cu media"),
            key="radio_metoda_valori_lipsa"
        )

        if metoda_selectata == "Completare cu mediana/moda":
            data_imputed = data.copy()
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    if data[col].dtype == 'object':
                        data_imputed[col] = data[col].fillna(data[col].mode()[0])
                    else:
                        data_imputed[col] = data[col].fillna(data[col].median())
            st.success("✅ Valorile lipsă au fost completate (mediana sau moda, după tipul coloanei)")
            st.dataframe(data_imputed.head())

        elif metoda_selectata == "Eliminare rânduri cu valori lipsă":
            data_dropped = data.dropna()
            st.warning(f"⚠️ S-au eliminat {len(data) - len(data_dropped)} rânduri care conțineau valori lipsă.")
            st.dataframe(data_dropped.head())

        elif metoda_selectata == "Completare cu media":
            data_mean_imputed = data.copy()
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    if data[col].dtype != 'object':
                        data_mean_imputed[col] = data[col].fillna(data[col].mean())
            st.success("✅ Valorile lipsă au fost completate cu media (doar coloane numerice).")
            st.dataframe(data_mean_imputed.head())


if section == "Analiza valorilor extreme":
    st.title("📊 Analiza Valorilor Extreme - Fashion Sales")

    st.markdown("""
    **Ce sunt valorile extreme?**  
    Valorile extreme sunt punctele de date care sunt semnificativ mai mari sau mai mici decât restul observațiilor. Acestea pot afecta analiza statistică și rezultatele modelului nostru, de aceea este important să le gestionăm corespunzător.

    **Cum le detectăm?**  
    Folosim metoda **Interquartile Range (IQR)**, care identifică valori extreme ca fiind cele care depășesc limita inferioară și superioară:

    - **Q1 (quartila 25%)** – valoarea sub care se află 25% dintre date
    - **Q3 (quartila 75%)** – valoarea sub care se află 75% dintre date
    - **IQR = Q3 - Q1**
    - **Limita inferioară** = Q1 - 1.5 * IQR
    - **Limita superioară** = Q3 + 1.5 * IQR

    Orice valoare în afara acestui interval este considerată **extremă**.
    """)
    sales_lb, sales_ub = detect_outliers_iqr(data, "Sales Volume")
    price_lb, price_ub = detect_outliers_iqr(data, "Price")

    st.write(f"🔹 **Limite pentru `Sales Volume`**: [{sales_lb:.2f}, {sales_ub:.2f}]")
    st.write(f"🔹 **Limite pentru `Price`**: [{price_lb:.2f}, {price_ub:.2f}]")

    #pastram datele
    var1 = data.copy()

    #eliminam valorile extreme
    var2 = data[(data["Sales Volume"] >= sales_lb) & (data["Sales Volume"] <= sales_ub) &
                (data["Price"] >= price_lb) & (data["Price"] <= price_ub)]

    #inlocuim valorile extreme cu percentila 95%
    var3 = data.copy()
    var3.loc[var3["Sales Volume"] > sales_ub, "Sales Volume"] = data["Sales Volume"].quantile(0.95)
    var3.loc[var3["Price"] > price_ub, "Price"] = data["Price"].quantile(0.95)

    option = st.radio(
        "Alege varianta de tratare a valorilor extreme:",
        ("Original", "Fara Valori Extreme", "Inlocuire cu Percentila 95%"),
        key="radio_analiza_valori_extreme"
    )

    if option == "Original":
        st.markdown("""
        **Varianta `Original` pastreaza toate datele așa cum sunt, fara a elimina sau modifica valori extreme.**
        Aceasta poate fi utila atunci cand vrem sa vedem intreaga distributie a datelor.
        """)
        st.pyplot(plot_boxplots(var1, "Original"))
        st.write(f"📊 Numar total de observații: {len(var1)}")

    elif option == "Fara Valori Extreme":
        st.markdown("""
        **Varianta `Fără Valori Extreme` elimină toate observațiile care depășesc limitele definite prin metoda IQR.**
        Aceasta este utilă dacă dorim să avem o distribuție mai curată a datelor, fără valori care ar putea distorsiona analiza.
        """)
        st.pyplot(plot_boxplots(var2, "Fără Valori Extreme"))
        st.write(f"📉 Număr de observații după eliminare: {len(var2)}")
        st.write(f"🚀 Observații eliminate: {len(var1) - len(var2)}")

    elif option == "Inlocuire cu Percentila 95%":
        st.markdown("""
        **Varianta `Înlocuire cu Percentila 95%` modifică valorile extreme peste limită cu valoarea percentilei 95%.**
        Aceasta este o metodă mai puțin agresivă decât eliminarea completă a datelor și poate fi utilă atunci când dorim să păstrăm informația generală.
        """)
        st.pyplot(plot_boxplots(var3, "Înlocuire cu Percentila 95%"))
        st.write(f"✅ Înlocuire realizată pentru valorile extreme peste percentila 95%")

    st.subheader("📊 Statistici descriptive")
    st.write("🔹 **Original**")
    st.write(var1[["Sales Volume", "Price"]].describe())

    if option != "Original":
        st.write("🔹 **Noua varianta de date**")
        if option == "Fără Valori Extreme":
            st.write(var2[["Sales Volume", "Price"]].describe())
        else:
            st.write(var3[["Sales Volume", "Price"]].describe())

if section == "Codificare și scalare":
    st.header("🧮 Codificare și scalare")

    categorical_cols = ['Brand', 'Category', 'Color']
    numeric_cols = ['Price', 'Sales Volume']

    #label encoding
    st.subheader("🌐 Codificare Label Encoding")
    label_encoded = data.copy()
    for col in categorical_cols:
        if col in label_encoded.columns:
            le = LabelEncoder()
            label_encoded[col] = le.fit_transform(label_encoded[col].astype(str))
    st.dataframe(label_encoded[categorical_cols].head())

    #one-hot encoding
    st.subheader("🔢 Codificare One-Hot Encoding")
    onehot_encoded = pd.get_dummies(data[categorical_cols])
    st.dataframe(onehot_encoded.head())

    #frequency encoding
    st.subheader("📊 Codificare Frequency Encoding")
    freq_encoded = data.copy()
    for col in categorical_cols:
        freq_map = freq_encoded[col].value_counts(normalize=True)
        freq_encoded[col + '_FreqEnc'] = freq_encoded[col].map(freq_map)
    st.dataframe(freq_encoded[[col + '_FreqEnc' for col in categorical_cols]].head())

    st.subheader("📏 Standardizare și Normalizare")
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    scaled_standard = scaler_standard.fit_transform(data[numeric_cols])
    scaled_minmax = scaler_minmax.fit_transform(data[numeric_cols])

    df_standard = pd.DataFrame(scaled_standard, columns=[f"{col}_std" for col in numeric_cols])
    df_minmax = pd.DataFrame(scaled_minmax, columns=[f"{col}_minmax" for col in numeric_cols])

    st.write("🔹 StandardScaler (media=0, std=1):")
    st.dataframe(df_standard.head())

    st.write("🔹 MinMaxScaler ([0,1]):")
    st.dataframe(df_minmax.head())

    st.subheader("📈 Vizualizări grafice")

    fig1, ax1 = plt.subplots()
    sns.histplot(data['Price'], bins=30, kde=True, ax=ax1)
    ax1.set_title('Distribuția Prețului')
    st.pyplot(fig1)
    st.markdown("""
     🔍 **Interpretare**: Prețurile sunt distribuite aproape uniform, ceea ce sugerează o strategie de preț variată în portofoliul de produse.
     """)

    fig2, ax2 = plt.subplots()
    sns.histplot(data['Sales Volume'], bins=30, kde=True, ax=ax2, color="orange")
    ax2.set_title('Distribuția Volumului de Vânzări')
    st.pyplot(fig2)
    st.markdown("""
     🔍 **Interpretare**: Vânzările sunt relativ echilibrate între produse, dar cu variații ușoare care pot reflecta sezonalitate sau preferințe.
     """)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Category', y='Price', data=data, ax=ax3)
    ax3.set_title('Boxplot Preț pe Categorii')
    st.pyplot(fig3)
    st.markdown("""
     🔍 **Interpretare**: Unele categorii (ex: Jewelry, Outerwear) tind să aibă prețuri mai ridicate, ceea ce reflectă poziționarea diferită pe piață.
     """)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='Color', y='Sales Volume', data=data, ax=ax4)
    ax4.set_title('Distribuția Vânzărilor pe Culoare (Violin plot)')
    st.pyplot(fig4)
    st.markdown("""
     🔍 **Interpretare**: Deși formele diferă, valorile medii ale vânzărilor sunt similare pe culori. Culorile influențează distribuția, dar nu drastic volumul total.
     """)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    corr = data[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title('Heatmap Corelații')
    st.pyplot(fig5)
    st.markdown("""
     🔍 **Interpretare**: Corelația dintre `Price` și `Sales Volume` este foarte mică (0.007), ceea ce sugerează că prețul nu influențează semnificativ volumul vânzărilor.
     """)
if section == "Identificare si stergere duplicate":
    duplicates = data.duplicated().sum()
    st.write(f"📌 Număr de duplicate: {duplicates}")
    if duplicates > 0:
        if st.button("Sterge duplicate"):
            data.drop_duplicates(inplace=True)
            st.success("✅ Duplicatele au fost sterse!")

if section == "Grupare dupa Categorie":
    sales_stats_by_category = data.groupby('Category')['Sales Volume'].agg(['sum', 'mean', 'min', 'max']).reset_index()
    price_stats_by_category = data.groupby('Category')['Price'].agg(['sum', 'mean', 'min', 'max']).reset_index()

    st.title("Analiza pe 'Categorie'")
    st.subheader("Statistici 'Sales Volume' per 'Categorie'")
    st.dataframe(sales_stats_by_category)
    st.subheader("Statistici 'Price' per 'Categorie'")
    st.dataframe(price_stats_by_category)

if section == "Predictie pret":
    st.header("🧠 Predicție Pret Produs cu ML")
    data['Price_log'] = np.log1p(data['Price'])

    df_encoded = pd.get_dummies(data.copy(), drop_first=True)
    target = 'Price_log'

    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    #random forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    #xgboots
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)

    st.subheader("📊 Rezultatele modelelor de regresie")

    st.markdown(f"""
    ### 🔹 Linear Regression
    - MAE: {lr_mae:.2f}  
    - MSE: {lr_mse:.2f}  
    - R²: {lr_r2:.3f}
    """)

    st.markdown(f"""
    ### 🌲 Random Forest
    - MAE: {rf_mae:.2f}  
    - MSE: {rf_mse:.2f}  
    - R²: {rf_r2:.3f}
    """)

    st.markdown(f"""
    ### ⚡ XGBoost
    - MAE: {xgb_mae:.2f}  
    - MSE: {xgb_mse:.2f}  
    - R²: {xgb_r2:.3f}
    """)

    st.success(
        "✅ Compararea modelelor este completă. Cel mai bun scor R² indică cel mai performant model pe datele actuale.")
    st.info("ℹ️ S-a aplicat transformarea logaritmică pe preț.")

    st.markdown("""
    ### 🧠 Interpretare rezultate:
    - 🔹 **Linear Regression** a obținut un scor R² de `0.931`, ceea ce înseamnă că aproximativ 93% din variația prețurilor este explicată de model. MAE-ul de `0.13` sugerează o eroare medie destul de mică, ceea ce e foarte bun pentru un model liniar de bază.

    - 🌲 **Random Forest** a obținut un scor R² excepțional de `0.999`, cu o eroare MAE de doar `0.01`. Acest lucru indică o potrivire aproape perfectă pe datele de testare, ceea ce ar putea fi un semn de **overfitting** (modelul a învățat prea bine datele).

    - ⚡ **XGBoost** are un scor R² de `0.998`, foarte apropiat de cel al Random Forest, cu aceeași eroare medie (`0.01`). Este un rezultat excelent, confirmând performanța ridicată a acestui model avansat.

    🔍 **Recomandare**:
    Atât **Random Forest** cât și **XGBoost** oferă performanțe foarte mari, dar aceste scoruri aproape perfecte pot fi înșelătoare. """)

if section == "Analiză Geografică":
    st.title("🌍 Vizualizare Geografică a Vânzărilor")

    world = gpd.read_file("dataIN/countries.geo.json")

    sales_by_country = data.groupby('Country')['Sales Volume'].sum().reset_index()
    geo_df = world.merge(sales_by_country, how='left', left_on='name', right_on='Country')

    fig, ax = plt.subplots(figsize=(15, 10))
    world.boundary.plot(ax=ax, linewidth=0.8, color='black')
    geo_df.plot(column='Sales Volume', ax=ax, legend=True, cmap='OrRd',
                missing_kwds={"color": "lightgrey", "label": "Fără date"})
    ax.set_title('Distribuția Vânzărilor pe Țări', fontsize=18)
    st.pyplot(fig)



    data["Region"] = data["Country"].apply(assign_region)
    region_stats = data.groupby('Region')['Sales Volume'].sum().reset_index()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Sales Volume', y='Region', data=region_stats, palette='coolwarm', ax=ax2)
    ax2.set_title("Vânzări totale pe regiuni")
    st.pyplot(fig2)

    st.markdown("""
    ### 🧠 Interpretare Vizualizare Geografică

    🌍 **Distribuția vânzărilor pe țări**:
    - Harta evidențiază țările în care s-au înregistrat cele mai mari volume de vânzări.
    - Țări precum **China**, **Statele Unite**, **Canada**, **Germania** și **Japonia** prezintă cele mai ridicate valori.
    - Culorile mai închise indică volume mai mari de vânzări, în timp ce zonele gri nu conțin date.

    📊 **Vânzări totale pe regiuni**:
    - **Europa de Vest** este liderul vânzărilor, cu un volum net superior altor regiuni.
    - **Asia** și **America** ocupă următoarele poziții, cu valori considerabile.
    - **Europa de Est** și zona etichetată „Altele” (ex: Australia) înregistrează volume mai scăzute.
    """)
if section == "Clusterizare clienți":
    st.header("👥 Segmentare clienți prin KMeans")
    df_kmeans = data[["Annual Income (k$)", "Spending Score"]].dropna()

    inertia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df_kmeans)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots()
    sns.lineplot(x=range(1, 11), y=inertia, marker='o', ax=ax)
    ax.set_title("Metoda cotului pentru alegerea numărului de clustere")
    st.pyplot(fig)

    n_clusters = st.slider("Alege numărul de clustere", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_kmeans["Cluster"] = kmeans.fit_predict(df_kmeans)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df_kmeans, x="Annual Income (k$)", y="Spending Score", hue="Cluster", palette="Set2", ax=ax2)
    ax2.set_title(f"Rezultatul clusterizării în {n_clusters} clustere")
    st.pyplot(fig2)

    st.markdown("""
    ### 🧠 Interpretare clusterizare KMeans

    📌 **Metoda cotului (Elbow Method):**
    - Graficul evidențiază punctul de inflexiune în jurul valorii `k=3`, ceea ce sugerează că 3 este un număr optim de clustere.
    - Alegerea a mai mult de 3 clustere duce la o îmbunătățire nesemnificativă a performanței și poate complica analiza.

    📊 **Rezultatul clusterizării în 3 clustere:**
    - Fiecare client a fost grupat în funcție de **Venitul Anual** și **Scorul de Cheltuieli**.
    - Clusterele rezultate sunt:
      - 🟢 **Cluster 0** – clienți cu scor mare de cheltuieli: potențiali cumpărători fideli sau segment VIP.
      - 🟠 **Cluster 1** – clienți cu scor mediu: cumpărători constanți, dar moderați.
      - 🔵 **Cluster 2** – clienți cu scor redus: cumpărători ocazionali, posibil sensibili la preț.

    📈 **Observații:**
    - Distribuția pe axe arată că scorul de cheltuieli este principalul diferențiator între clustere.
    - Venitul anual are o influență mai mică în separarea grupurilor, semnalând că atitudinea față de cheltuieli nu depinde direct de venit.


    """)


if section == "Clasificare logistică":
    st.header("🔐 Regresie logistică – clasificare clienți")

    data_class = data.copy()

    threshold = data_class['Spending Score'].quantile(0.75)
    data_class['HighSpender'] = (data_class['Spending Score'] > threshold).astype(int)

    unique_classes = data_class['HighSpender'].nunique()
    if unique_classes < 2:
        st.error("⚠️ Clasificarea logistică nu poate fi realizată: toate valorile din 'HighSpender' sunt identice.")
    else:
        features = pd.get_dummies(data_class[['Genre', 'Age', 'Annual Income (k$)']], drop_first=True)
        target = data_class['HighSpender']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        st.write("✅ Clasificarea a fost realizată cu succes.")


    st.write(f"🔹 Acuratețea modelului: {logreg.score(X_test, y_test):.2f}")
    st.markdown("""
    ### 🧠 Interpretare rezultate – Regresie Logistică

    🔍 Modelul de regresie logistică a fost antrenat pentru a **clasifica clienții în funcție de comportamentul lor de cumpărare**, mai exact dacă sunt sau nu „mari cheltuitori”.

    📊 **Acuratețea obținută: 0.75**
    - Acuratețea de `75%` înseamnă că modelul a clasificat corect 3 din 4 clienți din setul de test.
    - Este un rezultat **acceptabil**, indicând că modelul are o capacitate decentă de generalizare.

    """)

if section == "Regresie multiplă":
    st.header("📈 Regresie multiplă – analiză cu Statsmodels")

    df_stats = data.dropna()
    df_stats = pd.get_dummies(df_stats, drop_first=True)

    X = df_stats[["Age", "Annual Income (k$)", "Genre_Male"]]
    X = sm.add_constant(X)
    X = X.astype(float)

    y = df_stats["Spending Score"].astype(float)

    model = sm.OLS(y, X).fit()
    st.text(model.summary())


    st.markdown("""
    ### 📉 Interpretare regresie multiplă
    
    - 🔹 **R² = 0.000** → Modelul nu explică deloc variația scorului de cheltuieli.
    - 🔹 Niciuna dintre variabile (`Age`, `Annual Income`, `Genre_Male`) **nu este semnificativă** (p-valori > 0.05).
    - ⚠️ Modelul **nu este util** pentru predicție în forma actuală.
    - ✅ Se recomandă explorarea altor variabile sau metode de modelare.
    """)
