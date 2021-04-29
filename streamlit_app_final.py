import streamlit as st
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import plotly.express as px
import pickle
import numpy as np
import pydeck as pdk
import requests
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    LOC,
    ORG,
    NamesExtractor,
    DatesExtractor,
    AddrExtractor,

    Doc
)


def main():
    page = st.sidebar.selectbox("", ["Модуль ввода и обработки сообщения", "Модуль анализа накопленной информации"])
    page_bg_img = '''
    <style>
    
    .stApp {
      background-image: url("https://i.ibb.co/Xfn9CL7/qwer.jpg");
      background-size: cover;             
    }
    h1,h2,h3,h4,h5,h6,p {text-align:center; color: white;}
    
    </style><h1>Автоматизированная система интеллектуального анализа сообщений о преступлениях, правонарушениях и происшествиях</h1>
    '''
    st.markdown(page_bg_img,unsafe_allow_html=True)
  
   
    if page == "Модуль ввода и обработки сообщения":

        df = pd.read_excel("dd.xlsx")
        
        #Распознавание речи
        stt_button = Button(label="Голосовой ввод сообщения", width=150,button_type="primary")
        
        stt_button.js_on_event("button_click", CustomJS(code="""
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
         
            recognition.onresult = function (e) {
                var value = "";
                for (var i = e.resultIndex; i < e.results.length; ++i) {
                    if (e.results[i].isFinal) {
                        value += e.results[i][0].transcript;
                    }
                }
                if ( value != "") {
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                }
            }
            recognition.start();
            """))
        
        result = streamlit_bokeh_events(
            stt_button,
            events="GET_TEXT",
            key="listen",
            refresh_on_update=True,
            override_height=45,
            debounce_time=0)
        
        speech_text = ''
        
        if result:
            if "GET_TEXT" in result:
                speech_text = result.get("GET_TEXT")
        
        #загрузка файла обученной модели        
        file = "finalized_model2.pkl"
        fileobj = open(file, 'rb')
        model = pickle.load(fileobj)
        #st.markdown('''<style>h2 {text-align: center; color:red;}</style><h2>Модуль предварительной квалификации</h2>''',unsafe_allow_html=True)
        str_proba=''
        st.subheader("Сообщение о происшествии")
        text = st.text_area('',value=speech_text)
        
        if len(text)>1:
            proba = model.predict([text])
            str_proba = proba[0]
        
        st.subheader("Результат классификации")
        st.text_input('',value=str_proba)
        
        #извлечение именованных суцщностей
        segmenter = Segmenter()
        morph_vocab = MorphVocab()

        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb) 
        syntax_parser = NewsSyntaxParser(emb)
        ner_tagger = NewsNERTagger(emb)

        names_extractor = NamesExtractor(morph_vocab)
        dates_extractor = DatesExtractor(morph_vocab)
        addr_extractor = AddrExtractor(morph_vocab)
        
        doc = Doc(text)
        doc.segment(segmenter) 
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)

        doc.segment(segmenter)
        #(doc.tokens[:5])

        #(doc.sents[:5])

        doc.tag_morph(morph_tagger)
        #(doc.tokens[:5])
        for span in doc.spans:
            span.normalize(morph_vocab)

        #{_.text: _.normal for _ in doc.spans if _.text != _.normal}


        for token in doc.tokens:
            token.lemmatize(morph_vocab)
   
        #{_.text: _.lemma for _ in doc.tokens}

        matches = dates_extractor(text)
        facts = [i.fact.as_json for i in matches]
        
        data_extr = ''
        for f in facts:
            data_extr = (f"{f.get('day')}.{f.get('month')}.{f.get('year')}")
        
        #st.markdown('''<style>h2 {text-align: center; color:red;}</style><h2>Модуль извлечения именованных сущностей</h2>''',unsafe_allow_html=True)
        st.subheader("Извлеченная дата")
        st.text_input(" ",value=data_extr)
   
   
        fio = ''
        for span in doc.spans:
            if span.type == PER:
                span.extract_fact(names_extractor)

        names_dict = {_.normal: _.fact.as_dict for _ in doc.spans if _.fact}
        for key in names_dict:
            fio +=(' '+key)
        st.subheader("Извлеченная ФИО")   
        st.text_input("  ", value=fio)
               
        
        
        def fetch_coordinates(apikey, place):
            base_url = "https://geocode-maps.yandex.ru/1.x"
            params = {"geocode": place, "apikey": apikey, "format": "json"}
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            found_places = response.json()['response']['GeoObjectCollection']['featureMember']
            most_relevant = found_places[0]
            lon, lat = most_relevant['GeoObject']['Point']['pos'].split(" ")
            return lon, lat

        apikey = '103985a4-dff7-4b84-98ca-cdca115b7719'
    
        address_name = 'Воронеж, проспект Патриотов, 53'
        st.subheader("Адрес")
        address = st.text_input(" ", address_name)
            
        coords = fetch_coordinates(apikey, address)
        str0 = coords[0]
        str1 = coords[1]
        a,b = float(str0), float(str1)
        data_coords = np.array([[b,a]])
        
        df = pd.DataFrame(
        data_coords,
        columns=['lat', 'lon'])
        
        st.subheader("Интерактивная карта")
        st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/outdoors-v9',
        initial_view_state=pdk.ViewState(
        latitude=b,
        longitude=a,
        zoom=13,
        pitch=50,
        ),
        layers=[
        pdk.Layer(
        'HexagonLayer',
        data=df,
        get_position='[lon, lat]',
        radius=20,
        elevation_scale=4,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        ),
        pdk.Layer(
        'ScatterplotLayer',
        data=df,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=100,
        ),
        ],
        ))
        
        st.markdown('''<style>h2 {text-align: center; color:white;}</style><h2>Типовые сообщения для иллюстрации работы модуля ввода и обработки сообщений</h2>''',unsafe_allow_html=True)
        st.title("""Кража
        27 апреля 2021 года у Балашовой Анастасии Владимировны была похищена сумка,
        в которой находился кошелёк и личные вещи, неизвестным лицом в ТЦ Армада
        по адресу город Воронеж улица Героев Сибиряков,дом 65 А""")

        st.title("""Грабеж
        20 марта 2021 года неизвестный открыто похитил ключи 
        от автомобиля Toyota Mark2 у Неймана Виталия Сергеевича
        в кафе DOSKI по адресу город Воронеж ул. Плехановская, дом 9.""")
                    
        st.title("""Убийство
        1 апреля 2020 года в лесопарке Оптимистов найден неизвестный мужчина 
        без сознания с множественными ножевыми ранеиями""")
                  
        st.title("""Мошенничество
        28 января 2019 года неизвестный путем обмана 
        завладел 30000 рублей, принадлежащими ОАО""Юг""")
                  
        st.title("""Незаконный оборот наркотиков
        В ходе проверки 20 апреля 2021 Алексеева Андрея Ивановича
        по адресу Воронеж, ул. Кольцовская, дом 35, был обнаружен 
        белый порошок неизвестного происхождения в прозрачном пакете.""")         
                   
        st.title("""Подделка денег
        01.03.2018 В 11.45 НА УЛ.Беговая, Д.209, В ПАО "СБЕРБАНК", 
        при пересчете обнаружена и изъята купюра 5000 рублей № БХ 2742545,
        доставленная из ООО "АЛЬТА", УЛ. Космонавтов, Д.1А, с признаками подделки.""")
        
        st.title("""Тяжкий вред здоровью                                                      
        01.03.2018 на улице Профсоюзов города Воронеж неизвестный причинил черепно 
        мозговую травму Гачиной Валерии Александровне.""") 
                              
        st.title("""Средней тяжести вред здоровью                                                      
        20 февраля 2019 года неизвестный причинил телесные повреждения
        Косичкиной Виктории Витальевне в виде закрытого перелома правой кисти.""")   
   
    elif page == "Модуль анализа накопленной информации":
        df = pd.read_excel("dd.xlsx")
        dff = df[['Период','Категория', 'latitude', 'longitude']]
        cvs = ds.Canvas(plot_width=4000, plot_height=4000)
        
        aggs = cvs.points(dff, x='latitude', y='longitude')
        
        coords_lat, coords_lon = aggs.coords['latitude'].values, aggs.coords['longitude'].values
        
        coordinates = [[coords_lon[0], coords_lat[0]],
                       [coords_lon[-1], coords_lat[0]],
                       [coords_lon[-1], coords_lat[-1]],
                       [coords_lon[0], coords_lat[-1]]]
        col = [
                        [0.0, "red"],
                        [0.5, "red"],
                        [0.51111111, "yellow"],
                        [0.71111111, "yellow"],
                        [0.71111112, "red"],
                        [1, "red"]]
        
        calculation = st.multiselect("Категория:", ("Изготовление, хранение, перевозка или сбыт поддельных денег или ценных бумаг","Убийства",
                                                    "Незаконный оборот наркотических средств",
                                                    "Мошенничество",
                                                    "Разбой",
                                                    "Умышленное причинение средней тяжест вреда здоровью",
                                                    "Грабеж"))
        year = st.multiselect("Год:",('2016','2017','2018','2019'))
        
        
        if 'Изготовление, хранение, перевозка или сбыт поддельных денег или ценных бумаг' in calculation:
            df = pd.read_excel("dd.xlsx")
            dff = df[['Период','Категория', 'latitude', 'longitude']]
            dff = dff[dff['Категория'].isin(['поддел.бумаги'])]
            dff.dropna(subset=['latitude', 'longitude'], inplace=True)
            
            col = [
                        [0.0, "blue"],
                        [0.5, "blue"],
                        [0.51111111, "yellow"],
                        [0.71111111, "red"],
                        [0.71111112, "blue"],
                        [1, "blue"]]
                
        
        if 'Убийства' in calculation:
            df = pd.read_excel("dd.xlsx")
            dff = df[['Период','Категория', 'latitude', 'longitude']]
            dff = dff[dff['Категория'].isin(['Убийства'])]
            dff.dropna(subset=['latitude', 'longitude'], inplace=True)
            
        if 'Незаконный оборот наркотических средств' in calculation:
            df = pd.read_excel("dd.xlsx")
            dff = df[['Период','Категория', 'latitude', 'longitude']]
            dff = dff[dff['Категория'].isin(['Наркотики'])]
            dff.dropna(subset=['latitude', 'longitude'], inplace=True)
            
            
        if 'Разбой' in calculation :
            df = pd.read_excel("dd.xlsx")
            dff = df[['Год','Период','Категория', 'latitude', 'longitude']]
            dff = dff[dff['Категория'].isin(['Разбой'])]
            dff.dropna(subset=['latitude', 'longitude'], inplace=True)    
            if '2016' in year:
                df = pd.read_excel("dd.xlsx")
                dff = df[['Год','Период','Категория', 'latitude', 'longitude']]
                dff = dff[dff['Категория'].isin(['Разбой'])]
                dff = dff[dff['Год'].isin(['2016'])]
                dff.dropna(subset=['latitude', 'longitude'], inplace=True)
            if '2017' in year:
                df = pd.read_excel("dd.xlsx")
                dff = df[['Год','Период','Категория', 'latitude', 'longitude']]
                dff = dff[dff['Категория'].isin(['Разбой'])]
                dff = dff[dff['Год'].isin(['2017'])]
                dff.dropna(subset=['latitude', 'longitude'], inplace=True)
            
        if 'Умышленное причинение средней тяжест вреда здоровью' in calculation:
            df = pd.read_excel("dd.xlsx")
            dff = df[['Период','Категория', 'latitude', 'longitude']]
            dff = dff[dff['Категория'].isin(['ум. прич. ср.тяж. вреда зд.'])]
            dff.dropna(subset=['latitude', 'longitude'], inplace=True)  
            
        if 'Мошенничество' in calculation:
            df = pd.read_excel("dd.xlsx")
            dff = df[['Период','Категория', 'latitude', 'longitude']]
            dff = dff[dff['Категория'].isin(['Мошенничество'])]
            dff.dropna(subset=['latitude', 'longitude'], inplace=True)
            col = px.colors.sequential.Peach          
                
        if 'Грабеж' in calculation:
            df = pd.read_excel("dd.xlsx")
            dff = df[['Период','Категория', 'latitude', 'longitude']]
            dff = dff[dff['Категория'].isin(['Грабеж'])]
            dff.dropna(subset=['latitude', 'longitude'], inplace=True)
                
            if '2016' in year:
                df = pd.read_excel("dd.xlsx")
                dff = df[['Год','Период','Категория', 'latitude', 'longitude']]
                dff = dff[dff['Категория'].isin(['Грабеж'])]
                dff = dff[dff['Год'].isin(['2016'])]
                dff.dropna(subset=['latitude', 'longitude'], inplace=True)
            
        
        if '2016'  in year:
            fig = px.density_mapbox(dff, lat='latitude', lon='longitude', radius=25,animation_frame=("Период"),
                     color_continuous_scale= col,
                                opacity = 0.9
                                )
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, 
                              mapbox=dict(
                                  pitch=60,
                                  bearing=30
                              ))
            fig.update_layout(coloraxis_showscale=False)
            
            st.plotly_chart(fig, use_container_width=True)
        elif '2017'  in year:
             fig = px.density_mapbox(dff, lat='latitude', lon='longitude', radius=25,animation_frame=("Период"),animation_group="Период",
                     color_continuous_scale= col,
                                opacity = 0.9
                                )
             fig.update_layout(mapbox_style="open-street-map")
             fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, 
                              mapbox=dict(
                                  pitch=60,
                                  bearing=30
                              ))
             fig.update_layout(coloraxis_showscale=False)
            
             st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.density_mapbox(dff, lat='latitude', lon='longitude', radius=25,animation_group="Период",
                     color_continuous_scale= col,
                                opacity = 0.9
                                )
        
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, 
                              mapbox=dict(
                                  pitch=60,
                                  bearing=30
                              ))
            fig.update_layout(coloraxis_showscale=False)
            fig.update_layout(width=1000,height=600)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

