import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from core.gdc_downloader import GDCDataFetcher

DIRECTORIO_BASE = "/content/drive/MyDrive/data_tesis"
PROYECTO_GDC = "TCGA-KIRC"

jerarquia_gdc = {
    "Biospecimen": ["Slide Image", "Biospecimen Supplement"],
    "Clinical": ["Clinical Supplement", "Pathology Report"],
    "Transcriptome Profiling": ["Gene Expression Quantification", "miRNA Expression Quantification"],
    "Simple Nucleotide Variation": ["Masked Somatic Mutation"],
    "Copy Number Variation": ["Copy Number Segment", "Gene Level Copy Number"]
}

def render_ui():
    """
    Renders the interactive Colab interface for GDC Datasets extraction.
    """
    contenedor_principal = widgets.VBox()

    def mostrar_paso_1():
        titulo = widgets.HTML("<h3>Paso 1: Selecciona los requisitos de los pacientes (Multimodalidad)</h3>")
        diccionario_checks = {}
        hijos_acordeon = []
        titulos = []

        for categoria, tipos in jerarquia_gdc.items():
            checks = [widgets.Checkbox(description=t, value=False, indent=False) for t in tipos]
            diccionario_checks[categoria] = checks
            hijos_acordeon.append(widgets.VBox(checks))
            titulos.append(categoria)

        acordeon = widgets.Accordion(children=hijos_acordeon)
        for i, cat in enumerate(titulos):
            acordeon.set_title(i, cat)

        # Advanced options
        lbl_advanced = widgets.HTML("<h4>Opciones Avanzadas de Filtrado</h4>")
        txt_limit = widgets.IntText(value=10000, description="Límite descargas:", style={'description_width': 'initial'})
        txt_formats = widgets.Text(value="", placeholder="ej. BCR XML, TSV", description="Formatos exactos (coma sep):", style={'description_width': 'initial'})
        txt_manifest = widgets.Text(value="", placeholder="./manifest.json", description="Usar Manifiesto Previo (Ruta):", style={'description_width': 'initial'})
        
        opciones_avanzadas = widgets.VBox([lbl_advanced, txt_limit, txt_formats, txt_manifest])

        btn_siguiente = widgets.Button(description="Siguiente paso", button_style="info")
        mensaje_error = widgets.Output()

        def avanzar_paso_2(b):
            seleccionados = set()
            for checks in diccionario_checks.values():
                for chk in checks:
                    if chk.value:
                        seleccionados.add(chk.description)

            if not seleccionados:
                with mensaje_error:
                    clear_output()
                    print("Debes seleccionar al menos un requisito de dato para continuar.")
                return

            limit_val = txt_limit.value
            formats_val = [x.strip() for x in txt_formats.value.split(",")] if txt_formats.value.strip() else None
            manifest_val = txt_manifest.value.strip()

            mostrar_paso_2(seleccionados, limit_val, formats_val, manifest_val)

        btn_siguiente.on_click(avanzar_paso_2)
        contenedor_principal.children = [titulo, acordeon, opciones_avanzadas, btn_siguiente, mensaje_error]

    def mostrar_paso_2(tipos_permitidos, limit, data_formats, manifest_path):
        titulo = widgets.HTML(f"<h3>Paso 2: Descargar Data para {len(tipos_permitidos)} modalidades cruzadas</h3>")
        
        btn_descargar = widgets.Button(description="Procesar y Descargar", button_style="success", icon="download")
        btn_volver = widgets.Button(description="Volver al Paso 1", button_style="warning")
        panel_botones = widgets.HBox([btn_volver, btn_descargar])
        salida_descarga = widgets.Output()

        # Display config summary
        config_html = f"<ul><li><b>Límite:</b> {limit}</li>"
        if data_formats: config_html += f"<li><b>Formatos Permitidos:</b> {', '.join(data_formats)}</li>"
        if manifest_path: config_html += f"<li><b>Filtrando por cohort dict:</b> {manifest_path}</li>"
        config_html += "</ul>"
        
        summary_view = widgets.HTML(config_html)

        def descargar(b):
            with salida_descarga:
                clear_output()
                print(f"Iniciando orquestación con GDCDataFetcher para: {', '.join(tipos_permitidos)}...")
                try:
                    fetcher = GDCDataFetcher(PROYECTO_GDC)
                    
                    target_case_ids = None
                    if manifest_path:
                        print(f"Cargando ids desde el manifiesto: {manifest_path}")
                        target_case_ids = fetcher.load_cohort_manifest(manifest_path)
                        print(f"Interceptando {len(target_case_ids)} pacientes estrictos.")
                        
                    archivos_finales = fetcher.search_files(
                        data_types=list(tipos_permitidos),
                        data_formats=data_formats,
                        limit=limit,
                        target_case_ids=target_case_ids
                    )
                    
                    if not archivos_finales:
                        print("No se encontraron archivos que crucen con los requisitos.")
                        return
                        
                    fetcher.download_files(archivos_finales, DIRECTORIO_BASE)
                    
                    new_manifest_path = os.path.join(DIRECTORIO_BASE, "last_cohort_manifest.json")
                    fetcher.save_cohort_manifest(archivos_finales, new_manifest_path)
                    
                    print("\\n✅ Proceso completado exitosamente.")
                    
                except Exception as e:
                    print(f"❌ Error en la ejecución: {e}")

        def volver(b):
            mostrar_paso_1()

        btn_descargar.on_click(descargar)
        btn_volver.on_click(volver)

        contenedor_principal.children = [titulo, summary_view, panel_botones, salida_descarga]

    display(contenedor_principal)
    mostrar_paso_1()
