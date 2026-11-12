---
name: BurnScan Clinical System
colors:
  surface: '#f9f9fe'
  surface-dim: '#d9dade'
  surface-bright: '#f9f9fe'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f8'
  surface-container: '#ededf2'
  surface-container-high: '#e8e8ed'
  surface-container-highest: '#e2e2e7'
  on-surface: '#1a1c1f'
  on-surface-variant: '#414755'
  inverse-surface: '#2e3034'
  inverse-on-surface: '#f0f0f5'
  outline: '#717786'
  outline-variant: '#c1c6d7'
  surface-tint: '#005bc1'
  primary: '#0058bc'
  on-primary: '#ffffff'
  primary-container: '#0070eb'
  on-primary-container: '#fefcff'
  inverse-primary: '#adc6ff'
  secondary: '#006e28'
  on-secondary: '#ffffff'
  secondary-container: '#95f99d'
  on-secondary-container: '#00752b'
  tertiary: '#bb0011'
  on-tertiary: '#ffffff'
  tertiary-container: '#e6171e'
  on-tertiary-container: '#fffbff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a41'
  on-primary-fixed-variant: '#004493'
  secondary-fixed: '#95f99d'
  secondary-fixed-dim: '#79dc84'
  on-secondary-fixed: '#002107'
  on-secondary-fixed-variant: '#00531d'
  tertiary-fixed: '#ffdad6'
  tertiary-fixed-dim: '#ffb4ab'
  on-tertiary-fixed: '#410002'
  on-tertiary-fixed-variant: '#93000a'
  background: '#f9f9fe'
  on-background: '#1a1c1f'
  surface-variant: '#e2e2e7'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 38px
  headline-lg:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
  headline-md:
    fontFamily: Inter
    fontSize: 17px
    fontWeight: '600'
    lineHeight: 22px
  body-lg:
    fontFamily: Inter
    fontSize: 17px
    fontWeight: '400'
    lineHeight: 22px
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
  label-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  caption:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1rem
  margin: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system translates Apple Human Interface Guidelines (macOS and iPadOS) into an intensive emergency triage clinical interface specifically engineered for Vietnamese emergency departments. The brand posture is calm, hyper-legible, clinical, and completely free of decorative noise.

### Design Movement & Aesthetic
- **HIG Functional Precision**: Rooted in pure Apple HIG structural semantics—inset-grouped lists, structural toolbars, explicit primary/secondary contrast, and exact control ergonomics.
- **Selective Material Hierarchy**: Frosted glass materials (`rgba(249, 249, 249, 0.72)` with 24px blur and 180% saturation) are reserved strictly for navigation chrome and floating operation bars to ground them over the viewport. Content cards are flat, solid white (`#FFFFFF`) to ensure uncompromised contrast and instant cognitive processing under emergency room lighting.
- **Clinical Certainty**: Color is never decorative. State changes, burn depth classifications (Độ I, Độ II, Độ III), and triage alerts strictly enforce redundant visual signals: color tint, icon glyph, and textual indicator simultaneously.

### Language & Tone
- **Tone**: Authoritative, unobtrusive, objective, quiet.
- **Language**: Vietnamese (Tiếng Việt) with standardized medical nomenclature (e.g., *Diện tích bỏng TBSA*, *Chỉ số bù dịch Parkland*, *Mức độ tổn thương*).

## Colors

The palette reproduces Apple system tints with strict clinical utility. Neutral surfaces organize information without visual fatigue; specialized dark canvas surfaces are strictly isolated to burn photography inspection.

### Base Surfaces & Chrome
- **App Canvas**: `#F2F2F7` (System Gray 6 / Desktop Window Background). Flat, uniform, without gradient or glow.
- **Content Surfaces**: `#FFFFFF` (Pure White). Applied to cards, inspection modals, and inset groupings. Never translucent.
- **Medical Viewport Background**: `#1C1C1E` (Dark Neutral Tier 1). Exclusively dedicated to clinical photo frames, wound depth heatmaps, and photographic magnification viewers.
- **Separators / Borders**: `0.5px solid rgba(60, 60, 67, 0.29)` (macOS thin divider standard).
- **Control Fill**: `rgba(120, 120, 128, 0.12)`. Applied to passive segmented control tracks, search field pills, and secondary button baselines.

### Typography & Icon Colors
- **Nhãn chính (Primary Text)**: `#000000` (100% Black) for diagnostic headers, clinical values, and critical metrics.
- **Nhãn phụ (Secondary Text)**: `rgba(60, 60, 67, 0.60)` (`#3C3C4399`) for field descriptors, timestamps, and measurement units.
- **Nhãn chú thích (Tertiary Text)**: `rgba(60, 60, 67, 0.30)` (`#3C3C434D`) for disabled controls and placeholder text.

### Clinical Status & Burn Grading Palettes
Every status application must pair the solid tone (icon & text) with its respective tinted background fill (`0.12` alpha):
- **Độ I / Ổn định (Grade 1 / Stable)**:
  - Solid: `#248A3D` (Apple System Green accessible).
  - Tint Fill: `rgba(52, 199, 89, 0.12)`.
  - Icon: `checkmark.circle.fill` / Vòng tròn dấu kiểm.
- **Độ II / Thận trọng (Grade 2 / Warning & Partial Thickness)**:
  - Solid: `#C93400` (Apple System Orange accessible).
  - Tint Fill: `rgba(255, 149, 0, 0.12)`.
  - Icon: `exclamationmark.triangle.fill` / Tam giác cảnh báo.
- **Độ III / Nguy kịch (Grade 3 / Critical & Full Thickness)**:
  - Solid: `#D70015` (Apple System Red accessible).
  - Tint Fill: `rgba(255, 59, 48, 0.12)`.
  - Icon: `exclamationmark.octagon.fill` / Bát giác khẩn cấp.
- **Primary Accent / Tương tác**:
  - Solid: `#007AFF` (Apple System Blue) for primary actions, selection states, and interactive tabs.
  - Tint Fill: `rgba(0, 122, 255, 0.12)`.

## Typography

The typographic system maps directly to Apple's Human Interface Guidelines scale, optimized for Vietnamese tone marks (*dấu thanh*) with generous baseline tracking and clear distinction between glyph ascenders and diacritics.

### Font Hierarchy & Roles
- **Tiêu đề lớn (Large Title / `headline-xl`)**: `32px / 38px / Bold (700)`. Used exclusively for clinical case identification and master patient registry screens.
- **Tiêu đề phụ (Title 2 / `headline-lg`)**: `22px / 28px / Semibold (600)`. Designates primary structural sections (e.g., *Thông tin bệnh án*, *Đánh giá TBSA*).
- **Tiêu đề bảng & Thẻ (Headline / `headline-md`)**: `17px / 22px / Semibold (600)`. List section headers, modal titles, and vital metric card summaries.
- **Văn bản chuẩn (Body / `body-md`)**: `15px / 20px / Regular (400)`. Triage notes, diagnosis summaries, and medical instructions.
- **Nhãn phụ (Secondary / `body-sm`)**: `13px / 18px / Regular (400)`. Supporting metadata, input subtext, and audit timestamps.
- **Ghi chú (Caption / `caption`)**: `12px / 16px / Regular (400)`. Footnotes, device serials, and auxiliary unit legends.

### Numerical Data & Tabular Rules
All numerical figures, heart rate reads, blood pressure ratios, burn percentages, and fluid rates (ml/h) must render using `font-variant-numeric: tabular-nums` (monospaced figures) to avoid layout shifts during dynamic telemetry refreshes.

## Layout & Spacing

The application enforces a desktop web app form factor tailored for clinical monitors and iPadOS pointer workstations, fixed at a 1440px standard canvas width.

### Layout Architecture
- **Global Structure**: 3-column macOS split-view layout:
  - **Cột điều hướng (Primary Sidebar)**: 260px fixed width, flat `#F2F2F7` background with 0.5px right separator. Houses patient triage queues and status filters.
  - **Cột hình ảnh y khoa (Medical Viewport Stage)**: 680px flexible workspace with `#1C1C1E` dark stage background, framing high-resolution wound photography, thermal depth overlays, and 3D body map markers.
  - **Cột chẩn đoán & Chỉ số (Diagnostic Inspector)**: 500px fixed width, solid white surface housing calculated indices (Công thức Parkland, % TBSA, phân loại độ sâu tổn thương).
- **Margins & Gutters**: Outer margin `1.5rem` (24px); element separation gap `1rem` (16px).
- **Ergonomics**: All click/tap targets maintain an absolute minimum touch zone of `44px × 44px` to facilitate rapid triage input on bedside touch terminals and tablet carts.

## Elevation & Depth

This system avoids heavy synthetic shadows, using surface color divergence, delicate border outlines, and Apple material blurs to establish hierarchical elevation tiers.

### Materials & Translucency
1. **Light Window Chrome (Vật liệu kính sáng)**:
   - Tokens: `background: rgba(249, 249, 249, 0.72); backdrop-filter: blur(24px) saturate(180%); -webkit-backdrop-filter: blur(24px) saturate(180%);`
   - Border: `border-bottom: 0.5px solid rgba(60, 60, 67, 0.29);`
   - Scope: Header window bar, floating segmented toolbars, and the docked triage command bar at the base.
2. **Medical Viewer Glass (Vật liệu kính tối)**:
   - Tokens: `background: rgba(28, 28, 30, 0.55); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);`
   - Border: `border: 0.5px solid rgba(255, 255, 255, 0.15);`
   - Text/Icons: `#FFFFFF`
   - Scope: Strictly confined to floating HUD controls, zoom overlays, and measurement pins positioned on top of the `#1C1C1E` medical photo viewer.
3. **Card & Panel Elevation**:
   - Zero drop-shadows on content cards. Elevation is purely structural: solid `#FFFFFF` cards resting on the `#F2F2F7` window canvas.
   - Low-contrast hairline dividers (`0.5px solid rgba(60, 60, 67, 0.29)`) demarcate content boundaries.

## Shapes

Corner radii adhere directly to Apple platform metrics, reinforcing a soft yet disciplined physical continuity across UI components:

- **Điều khiển & Nút bấm (Controls & Buttons)**: `10px` (`roundedness` preset base). Standard for action buttons, segmented controls, and toolbar pills.
- **Trường nhập liệu & Danh sách (Inputs & Inset Groups)**: `12px` (`rounded-lg`). Enforces the classic iOS inset-grouped container shape.
- **Khung giao diện lớn (Large Inspection Panels & Modals)**: `16px` (`rounded-xl`). Applied to outer cards, clinical triage diagnostic panels, and photographic overlay stages.
- **Huy hiệu & Nhãn trạng thái (Status Chips & Badges)**: Fully rounded pill shape (`9999px`).

## Components

All components implement standard Apple interaction patterns tailored for high-stakes medical triage.

### 1. Inset-Grouped Lists (Danh sách nhóm dạng Inset)
- **Vỏ chứa (Container)**: Solid `#FFFFFF`, border-radius `12px`, padding `0`, no shadow.
- **Hàng (Row)**: Fixed height between `44px` and `52px`. Padding horizontal `16px`.
- **Đường phân cách (Dividers)**: `0.5px solid rgba(60, 60, 67, 0.29)`, inset from the left by `16px` (clearing left-aligned labels/icons), extending to the right edge. The final row has no divider.
- **Cấu trúc dữ liệu**:
  - Nhãn bên trái: `17px` Regular `#000000`.
  - Giá trị bên phải: `17px` Regular `rgba(60, 60, 67, 0.60)`.
  - Icon điều hướng (nếu có): `chevron.right`, `14px`, `rgba(60, 60, 67, 0.30)`.

### 2. Buttons (Nút tương tác)
- **Nút hành động chính (Primary Action CTA)**:
  - Background: Solid `#007AFF`.
  - Text: Solid `#FFFFFF`, `15px` Semibold.
  - Height: `44px`, border-radius `10px`.
  - State: Active/Pressed scales slightly to `0.98` with `0.85` opacity.
- **Nút phụ phân cấp (Tinted Secondary Button)**:
  - Background: `rgba(0, 122, 255, 0.12)`.
  - Text: Solid `#007AFF`, `15px` Semibold.
  - Height: `44px`, border-radius `10px`.
- **Nút khẩn cấp / Hủy (Destructive CTA)**:
  - Background: `rgba(255, 59, 48, 0.12)`.
  - Text: Solid `#D70015`, `15px` Semibold.

### 3. Apple HIG Segmented Control (Thanh chọn phân đoạn)
- **Đường ray (Track)**: Height `32px` (desktop compact) or `40px` (clinical touch mode), background `rgba(120, 120, 128, 0.12)`, border-radius `8px` to `10px`, padding `2px`.
- **Con trượt chọn (Active Thumb)**: Solid `#FFFFFF`, border-radius `7px` to `8px`, subtle shadow (`0px 3px 8px rgba(0,0,0,0.12), 0px 3px 1px rgba(0,0,0,0.04)`), text `#000000` Semibold `13px`.
- **Trạng thái không chọn (Inactive Tab)**: Background transparent, text `rgba(60, 60, 67, 0.60)` Regular `13px`.

### 4. Chips & Triage Badges (Huy hiệu phân loại bỏng)
- **Quy chuẩn hiển thị bắt buộc**: Color + SF Symbol Icon + Vietnamese Label.
- **Cấu trúc**: Height `28px`, border-radius `9999px`, horizontal padding `10px`, gap `6px`.
  - *Độ I (Grade 1)*: Background `rgba(52, 199, 89, 0.12)`, text `#248A3D`, icon `checkmark.circle.fill`. Label: "Độ I - Nông".
  - *Độ II (Grade 2)*: Background `rgba(255, 149, 0, 0.12)`, text `#C93400`, icon `exclamationmark.triangle.fill`. Label: "Độ II - Dày một phần".
  - *Độ III (Grade 3)*: Background `rgba(255, 59, 48, 0.12)`, text `#D70015`, icon `exclamationmark.octagon.fill`. Label: "Độ III - Toàn bộ lớp da".

### 5. Form Inputs (Trường nhập liệu)
- **Vùng nhập (Field Surface)**: Solid `#FFFFFF` inside inset groups, or `rgba(120, 120, 128, 0.12)` for standalone search bars. Height `44px`, border-radius `10px`.
- **Trạng thái Focus**: Zero fuzzy halo; sharp ring `2px solid #007AFF`.
- **Placeholder**: `rgba(60, 60, 67, 0.30)`.

### 6. Toggles & Sliders (Công tắc & Thanh trượt)
- **iOS Toggle Switch**: Width `51px`, height `31px`, track when off `rgba(120, 120, 128, 0.32)`, track when on `#34C759`. Thumb: Solid `#FFFFFF`, diameter `27px`, drop shadow.
- **Clinical Fluid/Burn Slider**: Track height `4px`, filled track `#007AFF`, remaining track `rgba(120, 120, 128, 0.20)`. Thumb: Solid `#FFFFFF`, diameter `28px`, elevation shadow, with visible numeric tooltip in tabular figures.

### 7. Clinical Alert Banners (Biểu ngữ cảnh báo lâm sàng)
- **Quy cách**: Banner cảnh báo không dùng kính (never glass). Sử dụng bề mặt màu đặc có sắc thái (solid tinted surface).
- **Cấu trúc**: Border-radius `12px`, padding `16px`, border `1px solid` tương ứng theo mức độ (Red/Orange).
- **Nội dung**: Icon lớn `20px` bên trái, tiêu đề in đậm `15px`, nội dung hướng dẫn cấp cứu `13px`.