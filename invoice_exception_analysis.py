#!/usr/bin/env python3
"""
Invoice Exception Analysis for SAP Data
Analyzes: RBKP (Invoice Verification), BSEG (Line Items), EKKO/EKPO (PO), 
BKPF (Accounting), KNA1/LFA1 (Master Data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/home/noirxvii/Downloads/archive (1)")

def load_csv_sample(filepath, nrows=10000):
    """Load CSV with error handling"""
    try:
        return pd.read_csv(filepath, nrows=nrows, low_memory=False)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_rbkp_invoice_verification():
    """Analyze RBKP - Invoice Verification headers"""
    print("=" * 60)
    print("RBKP - INVOICE VERIFICATION ANALYSIS")
    print("=" * 60)
    
    df = load_csv_sample(DATA_DIR / "rbkp.csv", 50000)
    if df is None:
        return
    
    print(f"\nTotal records loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Key fields for exceptions
    key_fields = ['mandt', 'belnr', 'gjahr', 'blart', 'bldat', 'budat', 'lifnr', 
                  'waers', 'rmwwr', 'zterm', 'xrech', 'rbstat', 'xmwst', 'reindat']
    print(f"\nKey fields present: {[f for f in key_fields if f in df.columns]}")
    
    # Document type distribution
    if 'blart' in df.columns:
        print("\n--- Document Types (BLART) ---")
        print(df['blart'].value_counts().head(10))
    
    # Invoice status (RBSTAT) - key for exceptions
    if 'rbstat' in df.columns:
        print("\n--- Invoice Status (RBSTAT) ---")
        print(df['rbstat'].value_counts())
        # RSTAT values: blank=open, X=completed, B=blocked, etc.
    
    # Payment block indicator
    if 'xrech' in df.columns:
        print("\n--- Payment Block (XRECH) ---")
        print(df['xrech'].value_counts())
    
    # Currency distribution
    if 'waers' in df.columns:
        print("\n--- Currencies ---")
        print(df['waers'].value_counts().head(10))
    
    # Vendors with most invoices
    if 'lifnr' in df.columns:
        print("\n--- Top Vendors by Invoice Count ---")
        print(df['lifnr'].value_counts().head(15))
    
    # Amount statistics
    if 'rmwwr' in df.columns:
        df['rmwwr_num'] = pd.to_numeric(df['rmwwr'], errors='coerce')
        print(f"\n--- Invoice Amount Stats ---")
        print(f"Mean: {df['rmwwr_num'].mean():.2f}")
        print(f"Median: {df['rmwwr_num'].median():.2f}")
        print(f"Max: {df['rmwwr_num'].max():.2f}")
        print(f"Zero/Null amounts: {(df['rmwwr_num'] <= 0).sum()}")
    
    # Date analysis
    if 'bldat' in df.columns:
        df['bldat_dt'] = pd.to_datetime(df['bldat'], errors='coerce')
        print(f"\n--- Date Range ---")
        print(f"Min: {df['bldat_dt'].min()}")
        print(f"Max: {df['bldat_dt'].max()}")
    
    if 'reindat' in df.columns:
        df['reindat_dt'] = pd.to_datetime(df['reindat'], errors='coerce')
        print(f"\n--- Receipt Date Range ---")
        print(f"Min: {df['reindat_dt'].min()}")
        print(f"Max: {df['reindat_dt'].max()}")
    
    # Overdue analysis
    if 'zterm' in df.columns and 'bldat' in df.columns and 'reindat' in df.columns:
        print("\n--- Payment Terms (ZTERM) ---")
        print(df['zterm'].value_counts().head(10))
    
    return df

def analyze_bseg_line_items():
    """Analyze BSEG - Accounting document line items"""
    print("\n" + "=" * 60)
    print("BSEG - LINE ITEM ANALYSIS")
    print("=" * 60)
    
    df = load_csv_sample(DATA_DIR / "bseg.csv", 50000)
    if df is None:
        return
    
    print(f"\nTotal records loaded: {len(df)}")
    
    # Key fields
    key_fields = ['bukrs', 'belnr', 'gjahr', 'buzei', 'shkzg', 'dmbtr', 'wrbtr',
                  'mwskz', 'saknr', 'hkont', 'kunnr', 'lifnr', 'ebeln', 'ebelp',
                  'matnr', 'werks', 'menge', 'meins', 'augdt', 'augbl']
    print(f"Key fields present: {[f for f in key_fields if f in df.columns]}")
    
    # Debit/Credit distribution
    if 'shkzg' in df.columns:
        print("\n--- Debit/Credit (SHKZG) ---")
        print(df['shkzg'].value_counts())
    
    # Account types
    if 'koart' in df.columns:
        print("\n--- Account Types (KOART) ---")
        print(df['koart'].value_counts())
    
    # Tax codes
    if 'mwskz' in df.columns:
        print("\n--- Tax Codes (MWSKZ) ---")
        print(df['mwskz'].value_counts().head(15))
    
    # Amount analysis
    if 'dmbtr' in df.columns:
        df['dmbtr_num'] = pd.to_numeric(df['dmbtr'], errors='coerce')
        print(f"\n--- Line Item Amounts ---")
        print(f"Mean: {df['dmbtr_num'].mean():.2f}")
        print(f"Negative amounts: {(df['dmbtr_num'] < 0).sum()}")
    
    # Clearing date analysis
    if 'augdt' in df.columns:
        df['augdt_dt'] = pd.to_datetime(df['augdt'], errors='coerce')
        cleared = df[df['augdt_dt'].notna()]
        print(f"\n--- Cleared Items: {len(cleared)} / {len(df)} ({len(cleared)/len(df)*100:.1f}%) ---")
    
    return df

def analyze_ekko_ekpo_purchase_orders():
    """Analyze EKKO/EKPO - Purchase Orders for 3-way matching"""
    print("\n" + "=" * 60)
    print("EKKO/EKPO - PURCHASE ORDER ANALYSIS")
    print("=" * 60)
    
    ekko = load_csv_sample(DATA_DIR / "ekko.csv", 20000)
    ekpo = load_csv_sample(DATA_DIR / "ekpo.csv", 50000)
    
    if ekko is not None:
        print(f"\nEKKO Records: {len(ekko)}")
        if 'bsart' in ekko.columns:
            print("\n--- PO Types (BSART) ---")
            print(ekko['bsart'].value_counts())
        if 'lifnr' in ekko.columns:
            print("\n--- Top Vendors in PO ---")
            print(ekko['lifnr'].value_counts().head(10))
    
    if ekpo is not None:
        print(f"\nEKPO Records: {len(ekpo)}")
        if 'netpr' in ekpo.columns:
            ekpo['netpr_num'] = pd.to_numeric(ekpo['netpr'], errors='coerce')
            print(f"\n--- PO Price Stats ---")
            print(f"Mean: {ekpo['netpr_num'].mean():.2f}")
        if 'pstyp' in ekpo.columns:
            print("\n--- Item Categories (PSTYP) ---")
            print(ekpo['pstyp'].value_counts())
    
    return ekko, ekpo

def analyze_master_data():
    """Analyze KNA1 (Customer) and LFA1 (Vendor) master data"""
    print("\n" + "=" * 60)
    print("MASTER DATA ANALYSIS")
    print("=" * 60)
    
    kna1 = load_csv_sample(DATA_DIR / "kna1.csv", 10000)
    lfa1 = load_csv_sample(DATA_DIR / "lfa1.csv", 10000)
    
    if kna1 is not None:
        print(f"\nKNA1 (Customers): {len(kna1)}")
        if 'land1' in kna1.columns:
            print("\n--- Customer Countries ---")
            print(kna1['land1'].value_counts().head(10))
        if 'ktokd' in kna1.columns:
            print("\n--- Customer Account Groups ---")
            print(kna1['ktokd'].value_counts())
        if 'sperr' in kna1.columns:
            print("\n--- Customer Blocks ---")
            print(kna1['sperr'].value_counts())
    
    if lfa1 is not None:
        print(f"\nLFA1 (Vendors): {len(lfa1)}")
        if 'land1' in lfa1.columns:
            print("\n--- Vendor Countries ---")
            print(lfa1['land1'].value_counts().head(10))
        if 'ktokk' in lfa1.columns:
            print("\n--- Vendor Account Groups ---")
            print(lfa1['ktokk'].value_counts())
        if 'sperr' in lfa1.columns:
            print("\n--- Vendor Blocks ---")
            print(lfa1['sperr'].value_counts())
        if 'zterm' in lfa1.columns:
            print("\n--- Vendor Payment Terms ---")
            print(lfa1['zterm'].value_counts().head(10))
    
    return kna1, lfa1

def detect_invoice_exceptions():
    """Main exception detection logic"""
    print("\n" + "=" * 60)
    print("INVOICE EXCEPTION DETECTION")
    print("=" * 60)
    
    # Load core tables
    rbkp = load_csv_sample(DATA_DIR / "rbkp.csv", 100000)
    bseg = load_csv_sample(DATA_DIR / "bseg.csv", 100000)
    ekko = load_csv_sample(DATA_DIR / "ekko.csv", 50000)
    ekpo = load_csv_sample(DATA_DIR / "ekpo.csv", 100000)
    lfa1 = load_csv_sample(DATA_DIR / "lfa1.csv", 10000)
    
    exceptions = []
    
    if rbkp is not None:
        # Exception 1: Blocked Invoices (RBSTAT = B or XRECH = X)
        if 'rbstat' in rbkp.columns:
            blocked = rbkp[rbkp['rbstat'].isin(['B', 'X'])]
            print(f"\n1. BLOCKED INVOICES (RBSTAT=B/X): {len(blocked)}")
            exceptions.append(('Blocked Invoice', blocked))
        
        if 'xrech' in rbkp.columns:
            pay_blocked = rbkp[rbkp['xrech'] == 'X']
            print(f"2. PAYMENT BLOCKED (XRECH=X): {len(pay_blocked)}")
            exceptions.append(('Payment Blocked', pay_blocked))
        
        # Exception 2: Invoices with zero/negative amounts
        if 'rmwwr' in rbkp.columns:
            rbkp['rmwwr_num'] = pd.to_numeric(rbkp['rmwwr'], errors='coerce')
            zero_amt = rbkp[rbkp['rmwwr_num'] <= 0]
            print(f"3. ZERO/NEGATIVE AMOUNT INVOICES: {len(zero_amt)}")
            exceptions.append(('Zero/Negative Amount', zero_amt))
        
        # Exception 3: Missing vendor master data
        if 'lifnr' in rbkp.columns and lfa1 is not None:
            rbkp_vendors = set(rbkp['lifnr'].astype(str).str.strip())
            lfa1_vendors = set(lfa1['lifnr'].astype(str).str.strip())
            missing_vendors = rbkp_vendors - lfa1_vendors
            print(f"4. INVOICES WITH MISSING VENDOR MASTER: {len(missing_vendors)} vendors")
            if missing_vendors:
                missing_df = rbkp[rbkp['lifnr'].astype(str).str.strip().isin(missing_vendors)]
                exceptions.append(('Missing Vendor Master', missing_df))
        
        # Exception 4: Overdue invoices (simplified)
        if 'bldat' in rbkp.columns and 'zterm' in rbkp.columns:
            rbkp['bldat_dt'] = pd.to_datetime(rbkp['bldat'], errors='coerce')
            # Assume standard terms: 30 days for now
            cutoff = datetime.now() - timedelta(days=90)  # 90 days old
            old_uncleared = rbkp[(rbkp['bldat_dt'] < cutoff) & (rbkp.get('rbstat', '') != 'X')]
            print(f"5. POTENTIALLY OVERDUE (>90 days, not completed): {len(old_uncleared)}")
            exceptions.append(('Potentially Overdue', old_uncleared))
        
        # Exception 5: Duplicate invoice numbers per vendor
        if 'lifnr' in rbkp.columns and 'xblnr' in rbkp.columns:
            duplicates = rbkp[rbkp.duplicated(subset=['lifnr', 'xblnr'], keep=False)]
            duplicates = duplicates[duplicates['xblnr'].notna() & (duplicates['xblnr'] != '')]
            print(f"6. DUPLICATE INVOICE REFERENCES (per vendor): {len(duplicates)}")
            exceptions.append(('Duplicate Invoice Reference', duplicates))
    
    # 3-Way Match Analysis: PO (EKKO/EKPO) -> GR (BSEG) -> Invoice (RBKP/BSEG)
    if ekpo is not None and bseg is not None:
        # Match PO line items to GR line items to Invoice line items
        if 'ebeln' in bseg.columns and 'ebelp' in bseg.columns:
            po_gr_match = bseg[bseg['ebeln'].notna() & bseg['ebelp'].notna()]
            print(f"\n7. BSEG LINES WITH PO REFERENCE: {len(po_gr_match)}")
            
            # Price variance: Compare BSEG amount to EKPO netpr
            if 'dmbtr' in po_gr_match.columns and 'netpr' in ekpo.columns:
                # This would require joining on ebeln/ebelp
                pass
    
    return exceptions

def analyze_vbrk_vbrp_billing():
    """Analyze VBRK/VBRP - Customer Billing Documents"""
    print("\n" + "=" * 60)
    print("VBRK/VBRP - BILLING DOCUMENT ANALYSIS")
    print("=" * 60)
    
    vbrk = load_csv_sample(DATA_DIR / "vbrk.csv", 20000)
    vbrp = load_csv_sample(DATA_DIR / "vbrp.csv", 50000)
    
    if vbrk is not None:
        print(f"\nVBRK Records: {len(vbrk)}")
        if 'fkart' in vbrk.columns:
            print("\n--- Billing Types (FKART) ---")
            print(vbrk['fkart'].value_counts())
        if 'fkdat' in vbrk.columns:
            vbrk['fkdat_dt'] = pd.to_datetime(vbrk['fkdat'], errors='coerce')
            print(f"\n--- Billing Date Range ---")
            print(f"Min: {vbrk['fkdat_dt'].min()}")
            print(f"Max: {vbrk['fkdat_dt'].max()}")
    
    return vbrk, vbrp

if __name__ == "__main__":
    print("SAP INVOICE EXCEPTION ANALYSIS")
    print("Data source:", DATA_DIR)
    
    # Run analyses
    rbkp = analyze_rbkp_invoice_verification()
    bseg = analyze_bseg_line_items()
    ekko, ekpo = analyze_ekko_ekpo_purchase_orders()
    kna1, lfa1 = analyze_master_data()
    vbrk, vbrp = analyze_vbrk_vbrp_billing()
    
    # Detect exceptions
    exceptions = detect_invoice_exceptions()
    
    print("\n" + "=" * 60)
    print("SUMMARY OF EXCEPTION TYPES FOUND")
    print("=" * 60)
    for exc_type, exc_df in exceptions:
        print(f"  - {exc_type}: {len(exc_df)} records")
    
    # Save exception summary
    summary = pd.DataFrame([{'exception_type': t, 'count': len(d)} for t, d in exceptions])
    summary.to_csv('/home/noirxvii/exception_summary.csv', index=False)
    print("\nException summary saved to /home/noirxvii/exception_summary.csv")