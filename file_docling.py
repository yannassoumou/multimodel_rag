from docling.document_converter import DocumentConverter

source = "./R2603/PDF/Bouton unique d'export.pdf"
converter = DocumentConverter()
doc = converter.convert(source).document
print(doc.export_to_markdown())