import pandas as pd
import numpy as np
import re
import logging
import streamlit as st
import io
from fuzzywuzzy import fuzz
import os
import glob


st.title("Scanning Tool")

# File uploaders
scanning_file = st.file_uploader("Upload Scannning CSV", type=["csv"])
catalog_file = st.file_uploader("Upload catalog CSV", type=["csv"])
pim_file = st.file_uploader("Upload PIM CSV", type=["csv"])
images_file = st.file_uploader("Upload Images CSV", type=["csv"])
SFTP_file = st.file_uploader("Upload SFTP CSV", type=["csv"])

# Run processing when all files are uploaded
if scanning_file and catalog_file and pim_file:
    st.success("All files uploaded. Ready to process!")

    if st.button("Run Processing"):
        # Load files into DataFrames
        df_scanning = pd.read_csv(scanning_file, dtype={"Barcode": str, "SKUs": str})
        df_sftp = pd.read_csv(SFTP_file, dtype={"sku": str})
        df_catalog = pd.read_csv(catalog_file)
        df_pim = pd.read_csv(pim_file)
        required_cols = ["master_product_code", "pieceBarcode", "productTitle::en", "productTitle::ar"]
        df_pim = df_pim[required_cols]
        df_images = pd.read_csv(images_file)

        # Format columns
        df_scanning['SKUs'] = df_scanning['SKUs'].astype(str).str.strip()
        df_sftp['sku'] = df_sftp['sku'].astype(str).str.strip()
        df_scanning['SKUs'] = df_scanning['SKUs'].replace(['nan', 'NaN', 'None'], '')
        df_scanning['Barcode'] = df_scanning['Barcode'].astype(str).str.strip()
        df_scanning['Barcode'] = df_scanning['Barcode'].replace(['nan', 'NaN', 'None'], '')
        missing_in_sftp = df_scanning[~df_scanning['SKUs'].isin(df_sftp['sku'])]

        df_catalog['pieceBarcode'] = df_catalog['pieceBarcode'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) else ''
        )

        # Rename columns in df_catalog
        df_catalog = df_catalog.rename(
            columns={
                "productName::en_EG": "productTitle::en_EG",
                "productName::ar_EG": "productTitle::ar_EG"
            }
        )

        # Normalize SKU formats
        df_scanning['SKUs'] = df_scanning['SKUs'].astype(str).str.strip().str.upper()
        df_catalog['sku'] = df_catalog['sku'].astype(str).str.strip().str.upper()

        # Vectorized cleaning
        df_scanning['ProductName'] = (
            df_scanning['ProductName']
            .astype(str)
            .str.replace(r'#P#', '', regex=True)
            .str.replace(r'#', '', regex=True)
            .str.replace(r'\(.*?\)', '', regex=True)
            .str.replace('-', '', regex=True)
            .str.replace('سعر جديد', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .str.title()
        )

        def check_missing_values(df, column_map):
            missing = {}
            for field in ['barcode', 'sku', 'price']:
                col = column_map.get(field)
                if col:
                    values = df[col].astype(str).str.strip()
                    if field == 'price':
                        invalid = ~values.str.replace('.', '', 1).str.isnumeric()
                    else:
                        invalid = values.isin(['', '.', '_'])
                    missing_rows = df[invalid]
                    if not missing_rows.empty:
                        label = f"Missing {field.capitalize()}s"
                        missing[label] = missing_rows
            return missing

        def infer_column_map(df):
            column_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'barcode' in col_lower and 'barcodes' not in column_map:
                    column_map['barcode'] = col
                elif 'sku' in col_lower and 'skus' not in column_map:
                    column_map['sku'] = col
                elif 'price' in col_lower:
                    column_map['price'] = col
            return column_map

        column_map = infer_column_map(df_scanning)
        missing_data = check_missing_values(df_scanning, column_map)

        # Identify invalid barcodes (Barcode == SKUs)
        df_scanning['is_invalid'] = (
            (df_scanning['Barcode'] == df_scanning['SKUs']) &
            (df_scanning['SKUs'].astype(str).str.len() < 7)
        )

        # Merge to check invalids in catalog
        potential_invalid_merged = pd.merge(
            df_scanning[df_scanning['is_invalid']],
            df_catalog[['pieceBarcode', 'sku']],
            left_on='Barcode',
            right_on='pieceBarcode',
            how='left'
        )

        invalid_barcodes = potential_invalid_merged[potential_invalid_merged['sku'].isna()].copy()
        invalid_indices = invalid_barcodes.index

        df_scanning.loc[invalid_indices, 'match_type'] = 'invalid_barcode'

        # Detect duplicate SKUs
        df_scanning['is_duplicate'] = df_scanning.duplicated(subset='SKUs', keep=False)

        # Detect duplicate Barcodes
        df_scanning['is_duplicate_barcode'] = df_scanning.duplicated(subset='Barcode', keep=False)

        # Merge on Barcode
        merge_barcode = pd.merge(
            df_scanning.copy(), df_catalog,
            left_on='Barcode', right_on='pieceBarcode',
            how='left', suffixes=('', '_cat')
        )

        matched_barcode = merge_barcode[~merge_barcode['sku'].isna()]
        unmatched_barcode = merge_barcode[merge_barcode['sku'].isna()]

        unmatched_barcode['SKUs'] = unmatched_barcode['SKUs'].astype(str)
        df_catalog['sku'] = df_catalog['sku'].astype(str)

        merge_SKUs = pd.merge(
            unmatched_barcode[df_scanning.columns], df_catalog,
            left_on='SKUs', right_on='sku',
            how='left', suffixes=('', '_cat')
        )

        merged_df = pd.concat([matched_barcode, merge_SKUs], ignore_index=True)

        conditions = [
            (merged_df['SKUs'] == merged_df['sku']) & (merged_df['Barcode'] != merged_df['pieceBarcode']),
            (merged_df['SKUs'] == merged_df['sku']),
            (merged_df['sku'].isna())
        ]
        choices = ['Change_Barcode', 'already_exist', 'new_item']
        merged_df['match_type'] = np.select(conditions, choices, default='switch_sku')

        final_df = pd.merge(
            df_scanning,
            merged_df[['Barcode', 'SKUs', 'sku', 'pieceBarcode', 'match_type']],
            on=['Barcode', 'SKUs'], how='left', suffixes=('', '_matched')
        )

        final_df['match_type'] = final_df['match_type'].combine_first(df_scanning['match_type'])

        final_df.loc[final_df['is_duplicate'] == True, 'match_type_matched'] = 'duplicate_sku'
        final_df.loc[final_df['is_invalid'] == True, 'match_type_matched'] = 'invalid_barcode'
        final_df.loc[final_df['is_duplicate_barcode'] == True, 'match_type_matched'] = 'duplicate_barcode'
        final_df.loc[
            (final_df['pieceBarcode'] != '') & (final_df['pieceBarcode'].notna()) & (final_df['is_invalid'] == True),
            'match_type_matched'
        ] = 'already_exist'

        # Prepare for merging with PIM
        df_pim['pieceBarcode'] = df_pim['pieceBarcode'].astype(str).str.replace(r'\s+', '', regex=True).str.strip()
        final_df['Barcode'] = final_df['Barcode'].str.replace(r'\s+', '', regex=True).str.strip()

        df_pim_exp = (
            df_pim[['pieceBarcode', 'productTitle::en', 'productTitle::ar', 'master_product_code']]
            .assign(pieceBarcode=df_pim['pieceBarcode'].str.split(','))
            .explode('pieceBarcode', ignore_index=True)
        )

        df_pim_exp['pieceBarcode'] = (
            df_pim_exp['pieceBarcode']
            .str.strip()
            .str.lstrip('0')
        )

        df_pim_exp['pieceBarcode'] = df_pim_exp['pieceBarcode'].astype('category')
        final_df['Barcode'] = final_df['Barcode'].astype('category')

        final_df = final_df.merge(
            df_pim_exp,
            left_on='Barcode', right_on='pieceBarcode',
            how='left'
        )

        final_df.loc[
            (final_df['match_type_matched'] == 'new_item') & (final_df['productTitle::en'].isna()),
            'match_type_matched'
        ] = 'not_on_pim'

        new_item = final_df[final_df['match_type_matched'] == 'new_item']

        mask = final_df['match_type_matched'] == 'already_exist'
        already_exist_rows = final_df[mask].copy()

        df_catalog_subset = df_catalog[['sku', 'pieceBarcode', 'productTitle::en_EG', 'master_product_code']].copy()
        df_catalog_subset['sku'] = df_catalog_subset['sku'].astype(str).str.strip()
        already_exist_rows['sku'] = already_exist_rows['sku'].astype(str).str.strip()

        df_catalog_subset = df_catalog_subset.drop_duplicates(subset=['sku'], keep='first')

        merged = already_exist_rows.merge(
            df_catalog_subset, on='sku', how='left', suffixes=('', '_from_catalog')
        ).set_index(already_exist_rows.index)

        final_df['pieceBarcode_y'] = final_df['pieceBarcode_y'].astype(str)
        final_df.loc[mask, 'pieceBarcode_y'] = merged['pieceBarcode'].astype(str)
        final_df.loc[mask, 'productTitle::en'] = merged['productTitle::en'].astype(str)
        final_df.loc[mask, 'master_product_code'] = merged['master_product_code'].astype(str)
        final_df['pieceBarcode_y'] = final_df['pieceBarcode_y'].astype(object)

        from sentence_transformers import SentenceTransformer, util

        with st.spinner("Loading semantic model and computing similarities..."):
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

            titles_1 = final_df['ProductName'].astype(str).tolist()
            titles_2 = final_df['productTitle::en'].astype(str).tolist()
            titles_3 = final_df['productTitle::ar'].astype(str).tolist()

            embeddings_1 = model.encode(titles_1, convert_to_tensor=True)
            embeddings_2 = model.encode(titles_2, convert_to_tensor=True)
            embeddings_3 = model.encode(titles_3, convert_to_tensor=True)

            cos_similarities = util.cos_sim(embeddings_1, embeddings_2).diagonal()
            cos_similarities2 = util.cos_sim(embeddings_1, embeddings_3).diagonal()

        final_df['semantic_similarity'] = (cos_similarities.cpu().numpy() * 100).round(0).astype(int)
        final_df['semantic_similarity_Arabic'] = (cos_similarities2.cpu().numpy() * 100).round(0).astype(int)
        final_df = final_df.drop_duplicates()
        new_item = new_item.drop_duplicates()

        new_item = new_item.merge(
            final_df[['ProductName', 'semantic_similarity', 'semantic_similarity_Arabic']],
            on='ProductName',
            how='left'
        )

        # Filter low similarity (both < 70)
        low_similarity = new_item[
            (new_item['semantic_similarity'] < 70) &
            (new_item['semantic_similarity_Arabic'] < 70)
        ]

        Override = low_similarity.copy()

        # Normalize Barcode format
        final_df['Barcode'] = final_df['Barcode'].astype(str).str.strip()
        df_images['Barcode'] = df_images['Barcode'].astype(str).str.strip()

        df_images_expanded = df_images.copy()
        df_images_expanded = df_images_expanded.assign(
            Barcode=df_images_expanded['Barcode'].str.split(',')
        ).explode('Barcode')
        df_images_expanded['Barcode'] = df_images_expanded['Barcode'].str.strip()

        not_on_pim = final_df[final_df['match_type_matched'] == 'not_on_pim'].copy()
        not_on_pim = not_on_pim.merge(
            df_images_expanded[['Barcode', 'Image', 'Title_EN', 'Title_AR']], on='Barcode', how='left'
        )
        not_on_pim.drop(
            columns=['is_invalid', 'match_type', 'is_duplicate', 'is_duplicate_barcode', 'sku',
                     'pieceBarcode_x', 'match_type_matched', 'pieceBarcode_y', 'productTitle::en', 'master_product_code'],
            inplace=True, errors='ignore'
        )

        Ready_Not_ON_Pim = not_on_pim[
            not_on_pim['Title_EN'].notna() & (not_on_pim['Title_EN'].str.strip() != '')
        ].copy()

        Not_On_Pim_Missing_Images = not_on_pim[
            not_on_pim['Title_EN'].isna() | (not_on_pim['Title_EN'].str.strip() == '')
        ].copy()

        override_new = new_item[new_item['match_type'] == 'Override'].copy()

        override_existing = final_df[
            (final_df['match_type_matched'] == 'already_exist') &
            (final_df['semantic_similarity'] < 45) &
            (final_df['semantic_similarity_Arabic'] < 45)
        ].copy()

        override = pd.concat([override_new, override_existing], ignore_index=True)
        override = override.merge(
            df_images_expanded[['Barcode', 'Image', 'Title_EN', 'Title_AR']], on='Barcode', how='left'
        )

        with st.spinner("Running second similarity pass on override items..."):
            model2 = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            titles_catalog_en = override['productTitle::en'].astype(str).tolist()
            titles_input_en = override['Title_EN'].astype(str).tolist()

            embeddings_catalog = model2.encode(titles_catalog_en, convert_to_tensor=True)
            embeddings_input = model2.encode(titles_input_en, convert_to_tensor=True)

            similarities = util.cos_sim(embeddings_catalog, embeddings_input).diagonal()

        override['final_similarity_check'] = (similarities.cpu().numpy() * 100).round(0).astype(int)
        override.drop(
            columns=['is_invalid', 'match_type', 'is_duplicate', 'is_duplicate_barcode', 'sku', 'pieceBarcode_x'],
            inplace=True, errors='ignore'
        )

        drop_condition = (
            (override['match_type_matched'] == 'already_exist') &
            (
                (override['final_similarity_check'] > 60) |
                (override['semantic_similarity'] > 60) |
                (override['semantic_similarity_Arabic'] > 60) |
                ((override['semantic_similarity'] + override['semantic_similarity_Arabic']) > 60)
            )
        )
        override = override[~drop_condition].copy()

        condition = (
            (override['match_type_matched'] == 'new_item') &
            (
                (override['semantic_similarity'] > 60) |
                (override['semantic_similarity_Arabic'] > 60) |
                (override['final_similarity_check'] > 60) |
                ((override['semantic_similarity'] + override['semantic_similarity_Arabic']) > 60)
            )
        )
        override = override[~condition].copy()

        # Restricted list
        restricted_keywords = [
            "tobacco",
            "nestle gallon",
            "nestle pure life water gallon, 18.9l",
            "nestle mineral water 18.9l",
            "nestle water bottle, 18.9l",
            "nestlé pure life water, 18.9l",
            "nestle pure life bottled drinking water, 20x330ml",
            "nestle pure life bottled drinking water, 20x600ml",
            "nestle pure life  water - 18.9l",
            "nestle pure life water gallon 18.9l (price of water without bottle exchange)",
            "boace turbos",
            "Mobile"
        ]
        restricted_keywords = [kw.lower().strip() for kw in restricted_keywords]

        def is_restricted(product_name: str) -> bool:
            name = str(product_name).lower()
            return any(kw in name for kw in restricted_keywords)

        new_item['is_restricted'] = new_item['productTitle::en'].apply(is_restricted)
        restricted_items = new_item[new_item['is_restricted']].copy()
        new_item = new_item[~new_item['is_restricted']].copy()

        # Pivot tables
        pivot_final_df = final_df['match_type_matched'].value_counts().reset_index()
        pivot_final_df.columns = ['match_type', 'count']

        pivot_new_item = new_item['match_type'].value_counts().reset_index()
        pivot_new_item.columns = ['match_type', 'count']

        combined_pivot = pd.concat([pivot_final_df, pivot_new_item], ignore_index=True)
        total_count = combined_pivot['count'].sum()
        total_row = pd.DataFrame([{'match_type': 'Total', 'count': total_count}])
        combined_pivot = pd.concat([combined_pivot, total_row], ignore_index=True)

        switch_df = final_df[
            (final_df['match_type_matched'] == 'switch_sku') & (final_df['semantic_similarity'] > 45)
        ]
        switch_df = switch_df.rename(columns={'SKUs': 'New_Sku', 'sku': 'Old_Sku'})

        duplicate_skus = final_df[
            (final_df['match_type_matched'] == 'duplicate_sku') &
            (final_df['SKUs'].notna()) &
            (final_df['SKUs'].astype(str).str.strip() != '')
        ]['SKUs'].unique()

        Gold_Minor = new_item[[
            "master_product_code",
            "SKUs",
            "Barcode",
            "productTitle::en"
        ]].rename(columns={
            "SKUs": "sku",
            "Barcode": "pieceBarcode"
        })

        Groups_excluded_df = final_df[final_df['SKUs'].astype(str).str.startswith(('316', '317'))]

        # Write Excel output
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            final_df.to_excel(writer, sheet_name='Final Output', index=False)
            new_item.to_excel(writer, sheet_name='New Items Breakdown', index=False)
            Gold_Minor.to_excel(writer, sheet_name='Gold_Minor', index=False)
            combined_pivot.to_excel(writer, sheet_name='Case_Comment', index=False)
            switch_df.to_excel(writer, sheet_name='Switch_Sku', index=False)
            Ready_Not_ON_Pim.to_excel(writer, sheet_name='Ready_Not_ON_Pim', index=False)
            Not_On_Pim_Missing_Images.to_excel(writer, sheet_name='Not_On_Pim_Missing_Images', index=False)
            override.to_excel(writer, sheet_name='override', index=False)
            missing_in_sftp.to_excel(writer, sheet_name='missing_in_sftp', index=False)
            restricted_items.to_excel(writer, sheet_name='restricted_items', index=False)
            Groups_excluded_df.to_excel(writer, sheet_name='Groups_excluded', index=False)

            workbook = writer.book
            ws_final = writer.sheets['Final Output']
            ws_new = writer.sheets['New Items Breakdown']

            orange_format = workbook.add_format({'bg_color': '#FFA500'})
            green_format = workbook.add_format({'bg_color': '#C6EFCE'})
            yellow_format = workbook.add_format({'bg_color': '#FFEB9C'})

            if 'match_type_matched' in final_df.columns:
                col_idx = final_df.columns.get_loc('match_type_matched')
                ws_final.set_column(col_idx, col_idx, None, orange_format)

            if 'semantic_similarity' in final_df.columns:
                col_idx = final_df.columns.get_loc('semantic_similarity')
                ws_final.set_column(col_idx, col_idx, None, green_format)

            if 'semantic_similarity' in new_item.columns:
                col_idx = new_item.columns.get_loc('semantic_similarity')
                ws_new.set_column(col_idx, col_idx, None, green_format)

            if 'match_type' in new_item.columns:
                col_idx = new_item.columns.get_loc('match_type')
                ws_new.set_column(col_idx, col_idx, None, yellow_format)

        output.seek(0)
        st.success("Processing complete!")
        st.dataframe(final_df.head(100))

        st.download_button(
            label="Download Final Excel Report",
            data=output,
            file_name="final_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.warning("Please upload all three CSV files.")
