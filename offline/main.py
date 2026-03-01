from utils.convert_pdf_image import download_pdf, pdf_to_images




def main():

    pdf_url = "https://climate.ec.europa.eu/system/files/2018-06/youth_magazine_en.pdf"
    pdf_path = download_pdf(pdf_url)
    document_images = pdf_to_images(pdf_path)

    document_images = document_images[4:10]

    queries = [
        {"text": "How much did the world temperature change so far?"},
        {"text": "What are the main causes of climate change?"},
    ]

    

if __name__ == "__main__":
    main()