#!/usr/bin/env bash
set -euo pipefail

# Download bibliography papers from methods.md Section 9:
# - PDF files
# - arXiv source tarballs (/src/<id>)
# Then:
# - extract all source tarballs
# - index .tex files and infer likely entrypoint per paper
#
# Usage:
#   bash run/download_papers.sh
# Optional envs:
#   METHODS_FILE=/abs/path/methods.md
#   OUT_DIR=/abs/path/papers/additional-bib
#   RETRIES=3
#   CONNECT_TIMEOUT=15

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
METHODS_FILE="${METHODS_FILE:-${ROOT_DIR}/methods.md}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/papers/additional-bib}"
RETRIES="${RETRIES:-3}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-15}"
CLEAN_FIRST="${CLEAN_FIRST:-1}"

PDF_DIR="${OUT_DIR}/pdf"
SRC_DIR="${OUT_DIR}/src"
EXTRACT_DIR="${OUT_DIR}/src_extracted"
META_DIR="${OUT_DIR}/meta"

ARXIV_LIST="${META_DIR}/arxiv_list.tsv"
OTHER_LIST="${META_DIR}/other_urls.tsv"
MANIFEST="${META_DIR}/download_manifest.tsv"
SRC_TEX_INDEX="${META_DIR}/src_tex_index.tsv"
SRC_TEX_FILES="${META_DIR}/src_tex_files.tsv"

mkdir -p "${PDF_DIR}" "${SRC_DIR}" "${EXTRACT_DIR}" "${META_DIR}"

if [[ ! -f "${METHODS_FILE}" ]]; then
  echo "[error] methods file not found: ${METHODS_FILE}" >&2
  exit 1
fi

if [[ "${CLEAN_FIRST}" == "1" ]]; then
  echo "[cleanup] clearing old files under ${OUT_DIR}"
  find "${PDF_DIR}" -maxdepth 1 -type f -exec rm -f {} +
  find "${SRC_DIR}" -maxdepth 1 -type f -name '*.tar.gz' -exec rm -f {} +
  find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

safe_name() {
  printf "%s" "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'
}

download_to() {
  local url="$1"
  local out="$2"
  curl -fL \
    --retry "${RETRIES}" \
    --retry-delay 1 \
    --connect-timeout "${CONNECT_TIMEOUT}" \
    "${url}" -o "${out}" >/dev/null 2>&1
}

extract_section9_lines() {
  awk '
    /^## 9\)/ {insec=1; next}
    insec && /^## / {insec=0}
    insec && /^- / {print}
  ' "${METHODS_FILE}"
}

extract_arxiv_id() {
  local url="$1"
  printf "%s\n" "${url}" | sed -nE 's#https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{5})(v[0-9]+)?#\1#p'
}

infer_entrypoint() {
  local paper_dir="$1"
  local entry=""
  local reason=""
  local docclass_only=""
  local first_tex=""

  if [[ -f "${paper_dir}/main.tex" ]]; then
    entry="${paper_dir}/main.tex"
    reason="main.tex_exists"
    printf "%s\t%s\n" "${entry}" "${reason}"
    return 0
  fi

  while IFS= read -r tex_file; do
    [[ -z "${tex_file}" ]] && continue
    [[ -z "${first_tex}" ]] && first_tex="${tex_file}"

    if rg -F -q '\documentclass' "${tex_file}"; then
      if rg -F -q '\begin{document}' "${tex_file}"; then
        entry="${tex_file}"
        reason="docclass_and_begin_document"
        printf "%s\t%s\n" "${entry}" "${reason}"
        return 0
      fi
      [[ -z "${docclass_only}" ]] && docclass_only="${tex_file}"
    fi
  done < <(find "${paper_dir}" -type f -name '*.tex' | sort)

  if [[ -n "${docclass_only}" ]]; then
    entry="${docclass_only}"
    reason="docclass_only"
    printf "%s\t%s\n" "${entry}" "${reason}"
    return 0
  fi

  if [[ -n "${first_tex}" ]]; then
    entry="${first_tex}"
    reason="first_tex_sorted"
    printf "%s\t%s\n" "${entry}" "${reason}"
    return 0
  fi

  printf "\tno_tex_files\n"
}

echo -e "name\tarxiv_id\turl" > "${ARXIV_LIST}"
echo -e "name\turl" > "${OTHER_LIST}"
echo -e "name\ttype\tid_or_url\tstatus\tbytes\tpath" > "${MANIFEST}"

echo "[download] parsing Section 9 from ${METHODS_FILE}"
line_count=0
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  line_count=$((line_count + 1))

  # Expected format:
  # - Paper Name: https://...
  name="$(printf "%s\n" "${line}" | sed -E 's/^- ([^:]+): .*/\1/')"
  url="$(printf "%s\n" "${line}" | sed -E 's/^- [^:]+: (https?:\/\/[^ ]+).*/\1/' | sed 's/[[:space:]]*$//')"
  paper_slug="$(safe_name "${name}")"

  if [[ -z "${name}" || -z "${url}" ]]; then
    echo "[warn] skip unparsable line: ${line}"
    continue
  fi

  arxiv_id="$(extract_arxiv_id "${url}")"
  if [[ -n "${arxiv_id}" ]]; then
    echo -e "${name}\t${arxiv_id}\t${url}" >> "${ARXIV_LIST}"
    pdf_path="${PDF_DIR}/${arxiv_id}__${paper_slug}.pdf"
    src_path="${SRC_DIR}/${arxiv_id}__${paper_slug}.tar.gz"

    pdf_status="ok"
    src_status="ok"
    if ! download_to "https://arxiv.org/pdf/${arxiv_id}.pdf" "${pdf_path}"; then
      pdf_status="fail"
      rm -f "${pdf_path}"
    fi
    if ! download_to "https://arxiv.org/src/${arxiv_id}" "${src_path}"; then
      src_status="fail"
      rm -f "${src_path}"
    fi

    pdf_bytes=0
    src_bytes=0
    [[ -f "${pdf_path}" ]] && pdf_bytes="$(wc -c < "${pdf_path}" | tr -d ' ')"
    [[ -f "${src_path}" ]] && src_bytes="$(wc -c < "${src_path}" | tr -d ' ')"

    echo -e "${name}\tpdf\t${arxiv_id}\t${pdf_status}\t${pdf_bytes}\t${pdf_path}" >> "${MANIFEST}"
    echo -e "${name}\tsrc\t${arxiv_id}\t${src_status}\t${src_bytes}\t${src_path}" >> "${MANIFEST}"
    continue
  fi

  echo -e "${name}\t${url}" >> "${OTHER_LIST}"

  # Fallback downloader for non-arXiv pages (e.g., IJCAI proceedings)
  html_cache="${META_DIR}/${paper_slug}.html"
  page_status="ok"
  pdf_status="fail_no_pdf_link"
  pdf_path=""
  pdf_bytes=0

  if download_to "${url}" "${html_cache}"; then
    pdf_link="$(rg -o "https?://[^\"']+\\.pdf|/proceedings/[0-9]{4}/[^\"']+\\.pdf" "${html_cache}" | head -n1 || true)"
    if [[ -n "${pdf_link}" ]]; then
      if [[ "${pdf_link}" == /* ]]; then
        pdf_link="https://www.ijcai.org${pdf_link}"
      fi
      pdf_path="${PDF_DIR}/${paper_slug}.pdf"
      if download_to "${pdf_link}" "${pdf_path}"; then
        pdf_status="ok"
        pdf_bytes="$(wc -c < "${pdf_path}" | tr -d ' ')"
      else
        pdf_status="fail_pdf"
        rm -f "${pdf_path}"
        pdf_path=""
      fi
    fi
  else
    page_status="fail_page"
  fi

  if [[ "${page_status}" != "ok" ]]; then
    echo -e "${name}\tpage\t${url}\t${page_status}\t0\t${html_cache}" >> "${MANIFEST}"
  fi
  echo -e "${name}\tpdf\t${url}\t${pdf_status}\t${pdf_bytes}\t${pdf_path}" >> "${MANIFEST}"
done < <(extract_section9_lines)

if [[ "${line_count}" -eq 0 ]]; then
  echo "[error] no bibliography lines found under '## 9)' in ${METHODS_FILE}" >&2
  exit 1
fi

echo "[extract] unpacking arXiv sources to ${EXTRACT_DIR}"
while IFS= read -r tgz_file; do
  [[ -z "${tgz_file}" ]] && continue
  base="$(basename "${tgz_file}" .tar.gz)"
  dst="${EXTRACT_DIR}/${base}"
  mkdir -p "${dst}"
  if ! tar -xzf "${tgz_file}" -C "${dst}" >/dev/null 2>&1; then
    echo "[warn] failed to extract ${tgz_file}" >&2
  fi
done < <(find "${SRC_DIR}" -maxdepth 1 -type f -name '*.tar.gz' | sort)

echo -e "name\tarxiv_id\textracted_dir\ttex_count\tentrypoint\treason" > "${SRC_TEX_INDEX}"
echo -e "name\tarxiv_id\ttex_file" > "${SRC_TEX_FILES}"

echo "[index] indexing .tex files and likely entrypoints"
while IFS=$'\t' read -r name arxiv_id url; do
  [[ "${name}" == "name" ]] && continue
  dir_pattern="${EXTRACT_DIR}/${arxiv_id}__*"
  paper_dir="$(find ${dir_pattern} -maxdepth 0 -type d 2>/dev/null | head -n1 || true)"
  if [[ -z "${paper_dir}" ]]; then
    echo -e "${name}\t${arxiv_id}\t\t0\t\tmissing_extracted_dir" >> "${SRC_TEX_INDEX}"
    continue
  fi

  tex_count="$(find "${paper_dir}" -type f -name '*.tex' | wc -l | tr -d ' ')"
  if [[ "${tex_count}" -gt 0 ]]; then
    while IFS= read -r tex_file; do
      rel="${tex_file#${ROOT_DIR}/}"
      echo -e "${name}\t${arxiv_id}\t${rel}" >> "${SRC_TEX_FILES}"
    done < <(find "${paper_dir}" -type f -name '*.tex' | sort)
  fi

  entry_raw="$(infer_entrypoint "${paper_dir}")"
  entrypoint="$(printf "%s" "${entry_raw}" | cut -f1)"
  reason="$(printf "%s" "${entry_raw}" | cut -f2)"
  entry_rel="${entrypoint#${ROOT_DIR}/}"
  dir_rel="${paper_dir#${ROOT_DIR}/}"

  echo -e "${name}\t${arxiv_id}\t${dir_rel}\t${tex_count}\t${entry_rel}\t${reason}" >> "${SRC_TEX_INDEX}"
done < "${ARXIV_LIST}"

pdf_count="$(find "${PDF_DIR}" -maxdepth 1 -type f | wc -l | tr -d ' ')"
src_count="$(find "${SRC_DIR}" -maxdepth 1 -type f -name '*.tar.gz' | wc -l | tr -d ' ')"
extracted_count="$(find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"

echo "[done] pdf_files=${pdf_count} src_files=${src_count} extracted_dirs=${extracted_count}"
echo "[done] manifest: ${MANIFEST}"
echo "[done] tex index: ${SRC_TEX_INDEX}"
