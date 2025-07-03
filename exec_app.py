import streamlit.web.cli

if __name__ == '__main__':

    flag_options = {'server.port': 8506,
                    'global.developmentMode': False}

    streamlit.web.cli.bootstrap.load_config_options(flag_options=flag_options)

    streamlit.web.cli.bootstrap.run(

        'pagina_inicial.py',
        True,
        [],
        flag_options
    )