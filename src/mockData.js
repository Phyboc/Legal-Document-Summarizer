// Language definitions supported by Legal Lens
export const LANGUAGES = [
  { code: 'en', label: 'English', native: 'English' },
  { code: 'te', label: 'Telugu', native: 'తెలుగు' },
  { code: 'hi', label: 'Hindi', native: 'हिन्दी' },
  { code: 'ta', label: 'Tamil', native: 'தமிழ்' },
  { code: 'kn', label: 'Kannada', native: 'ಕನ್ನಡ' },
  { code: 'ml', label: 'Malayalam', native: 'മലയാളം' },
  { code: 'mr', label: 'Marathi', native: 'मराठी' },
  { code: 'bn', label: 'Bengali', native: 'বাংলা' }
];

// Step messages for the loading state pulse animation
export const LOADING_STEPS = [
  "Uploading legal document to secure sandbox...",
  "Running optical character recognition (OCR) on scans...",
  "Parsing complex legal terminology and definitions...",
  "Applying Legal Lens AI semantic models...",
  "Extracting key clauses and obligation variables...",
  "Generating simple summaries and translating definitions..."
];

// Mock database for document translations
export const MOCK_DOCUMENTS = [
  {
    id: 'rental_agreement',
    name: 'Residential Rental Agreement',
    filename: 'Standard_Residential_Rental_Agreement.pdf',
    size: '245 KB',
    data: {
      en: {
        summary: "An 11-month lease agreement between landlord Rajesh Sharma and tenant Amit Verma for Apartment 4B, Green Meadows. Rent is ₹25,000/month, security deposit is ₹1,00,000, and there is a 10% rent increment upon renewal. Minor repairs under ₹2,000 are the tenant's responsibility.",
        clauses: [
          {
            title: "Security Deposit Clause",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "You pay ₹1,00,000 upfront. The landlord must return this interest-free deposit when you move out, minus any money needed for repairs beyond normal wear and tear."
          },
          {
            title: "Rent Escalation",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "If you choose to stay past 11 months and renew the lease, your monthly rent will automatically go up by 10% (increasing from ₹25,000 to ₹27,500)."
          },
          {
            title: "Maintenance and Repairs",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "You must pay for minor fixes (like lightbulbs, leaking taps, or switch issues) if they cost less than ₹2,000. Big structural repairs are the landlord's responsibility."
          }
        ]
      },
      te: {
        summary: "ఇది యజమాని రాజేష్ శర్మ మరియు అద్దెదారు అమిత్ వర్మ మధ్య గ్రీన్ మెడోస్ అపార్ట్‌మెంట్ 4B కోసం జరిగిన 11 నెలల అద్దె ఒప్పందం. అద్దె నెలకు ₹25,000, సెక్యూరిటీ డిపాజిట్ ₹1,00,000, మరియు పునరుద్ధరణ సమయంలో 10% అద్దె పెంపు ఉంటుంది. ₹2,000 లోపు చిన్న మరమ్మతులు అద్దెదారు బాధ్యత.",
        clauses: [
          {
            title: "సెక్యూరిటీ డిపాజిట్ నిబంధన",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "మీరు ₹1,00,000 అడ్వాన్స్ కట్టాలి. ఇల్లు ఖాళీ చేసేటప్పుడు, ఏవైనా నష్టాల ఖర్చులు మినహాయించి యజమాని ఈ వడ్డీ లేని డబ్బును మీకు తిరిగి ఇవ్వాలి."
          },
          {
            title: "అద్దె పెంపు",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "11 నెలల తర్వాత మీరు అద్దె ఒప్పందాన్ని పొడిగిస్తే, మీ నెలవారీ అద్దె 10% పెరుగుతుంది (అంటే ₹25,000 నుండి ₹27,500 అవుతుంది)."
          },
          {
            title: "నిర్వహణ మరియు మరమ్మతులు",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "₹2,000 లోపు ఉండే చిన్న రిపేర్లు (లైట్లు లేదా నల్లాలు మార్చడం వంటివి) మీరే చేయించుకోవాలి. పెద్ద రిపేర్లకు యజమానే ఖర్చు భరిస్తారు."
          }
        ]
      },
      hi: {
        summary: "मकान मालिक राजेश शर्मा और किरायेदार अमित वर्मा के बीच अपार्टमेंट 4B, ग्रीन मीडोज के लिए 11 महीने का लीज समझौता। किराया ₹25,000/माह है, सुरक्षा जमा ₹1,00,000 है, और नवीनीकरण पर 10% किराया वृद्धि है। ₹2,000 से कम की मामूली मरम्मत किरायेदार की जिम्मेदारी है।",
        clauses: [
          {
            title: "सुरक्षा जमा (सिक्योरिटी डिपॉजिट) क्लॉज",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "आपको ₹1,00,000 सुरक्षा जमा (सिक्योरिटी डिपॉजिट) देना होगा। जब आप घर खाली करेंगे, तो मकान मालिक नुकसान की कटौती (यदि कोई हो) करके इसे वापस कर देगा।"
          },
          {
            title: "किराया वृद्धि",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "यदि आप 11 महीने के बाद अनुबंध का नवीनीकरण करते हैं, तो आपका किराया 10% बढ़ जाएगा (₹25,000 से बढ़कर ₹27,500 हो जाएगा)।"
          },
          {
            title: "रखरखाव और मरम्मत",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "₹2,000 से कम के छोटे-मोटे खर्च (जैसे नल या लाइट ठीक कराना) आपको खुद उठाने होंगे। बड़े काम मकान मालिक कराएगा।"
          }
        ]
      },
      ta: {
        summary: "வீட்டு உரிமையாளர் ராஜேஷ் சர்மா மற்றும் வாடகைதாரர் அமித் வர்மா இடையே கிரீன் மெடோஸ் அபார்ட்மெண்ட் 4B க்கான 11 மாத வாடகை ஒப்பந்தம். வாடகை மாதம் ₹25,000, பாதுகாப்பு வைப்புத்தொகை ₹1,00,000 மற்றும் புதுப்பிக்கும் போது 10% வாடகை உயர்வு இருக்கும். ₹2,000 க்கும் குறைவான சிறிய பழுதுபார்ப்புகள் வாடகைதாரரின் பொறுப்பாகும்.",
        clauses: [
          {
            title: "பாதுகாப்பு வைப்புத்தொகை பிரிவு",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "நீங்கள் ₹1,00,000 முன்பணமாக செலுத்த வேண்டும். நீங்கள் வீட்டை காலி செய்யும்போது, சேதங்கள் ஏதும் இருப்பின் அதற்கான தொகையை கழித்துக் கொண்டு உரிமையாளர் இதை வட்டி இன்றி திருப்பித் தருவார்."
          },
          {
            title: "வாடகை உயர்வு",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "11 மாதங்களுக்குப் பிறகு ஒப்பந்தத்தை புதுப்பித்தால், உங்கள் வாடகை 10% அதிகரிக்கும் (அதாவது ₹25,000 லிருந்து ₹27,500 ஆக உயரும்)."
          },
          {
            title: "பராமரிப்பு மற்றும் பழுதுபார்ப்பு",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "₹2,000 க்கும் குறைவான சிறிய பழுதுபார்ப்புகளை (பல்பு அல்லது குழாய் சரிசெய்தல்) நீங்களே செய்ய வேண்டும். பெரிய பழுதுகளுக்கு உரிமையாளரே பொறுப்பு."
          }
        ]
      },
      kn: {
        summary: "ಮನೆ ಮಾಲೀಕ ರಾಜೇಶ್ ಶರ್ಮಾ ಮತ್ತು ಬಾಡಿಗೆದಾರ ಅಮಿತ್ ವರ್ಮಾ ನಡುವೆ ಗ್ರೀನ್ ಮೆಡೋಸ್ ಅಪಾರ್ಟ್‌ಮೆಂಟ್ 4B ಗಾಗಿ 11 ತಿಂಗಳ ಬಾಡಿಗೆ ಒಪ್ಪಂದ. ಬಾಡಿಗೆ ತಿಂಗಳಿಗೆ ₹25,000, ಸೆಕ್ಯುರಿಟಿ ಡೆಪಾಸಿಟ್ ₹1,00,000, ಮತ್ತು ನವೀಕರಣದ ಸಮಯದಲ್ಲಿ 10% ಬಾಡಿಗೆ ಹೆಚ್ಚಳವಿರುತ್ತದೆ. ₹2,000 ಒಳಗಿನ ಸಣ್ಣ ರಿಪೇರಿಗಳು ಬಾಡಿಗೆದಾರರ ಜವಾಬ್ದಾರಿಯಾಗಿದೆ.",
        clauses: [
          {
            title: "ಸೆಕ್ಯುರಿಟಿ ಡೆಪಾಸಿಟ್ ಷರತ್ತು",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "ನೀವು ₹1,00,000 ಮುಂಗಡ ಹಣ ನೀಡಬೇಕು. ಮನೆ ಖಾಲಿ ಮಾಡುವಾಗ ಹಾನಿ ವೆಚ್ಚಗಳನ್ನು ಕಡಿತಗೊಳಿಸಿ ಮಾಲೀಕರು ಈ ಬಡ್ಡಿ ರಹಿತ ಹಣವನ್ನು ನಿಮಗೆ ಹಿಂತಿರುಗಿಸಬೇಕು."
          },
          {
            title: "ಬಾಡಿಗೆ ಹೆಚ್ಚಳ",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "11 ತಿಂಗಳ ನಂತರ ನೀವು ಒಪ್ಪಂದವನ್ನು ನವೀಕರಿಸಿದರೆ, ಬಾಡಿಗೆ 10% ರಷ್ಟು ಹೆಚ್ಚಾಗುತ್ತದೆ (ಅಂದರೆ ₹25,000 ರಿಂದ ₹27,500 ಆಗುತ್ತದೆ)."
          },
          {
            title: "ನಿರ್ವಹಣೆ ಮತ್ತು ದುರಸ್ತಿ",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "₹2,000 ಒಳಗಿನ ಸಣ್ಣಪುಟ್ಟ ರಿಪೇರಿಗಳನ್ನು (ಲೈಟ್ ಬಲ್ಬ್ ಅಥವಾ ನಲ್ಲಿ ದುರಸ್ತಿ) ನೀವೇ ಮಾಡಿಸಬೇಕು. ದೊಡ್ಡ ರಿಪೇರಿ ವೆಚ್ಚವನ್ನು ಮಾಲೀಕರೇ ಭರಿಸುತ್ತಾರೆ."
          }
        ]
      },
      ml: {
        summary: "കെട്ടിട ഉടമ രാജേഷ് ശർമ്മയും വാടകക്കാരൻ അമിത് വർമ്മയും തമ്മിലുള്ള 11 മാസത്തെ വാടക കരാർ. വാടക മാസം ₹25,000, സെക്യൂരിറ്റി ഡെപ്പോസിറ്റ് ₹1,00,000, പുതുക്കുമ്പോൾ 10% വാടക വർദ്ധനവ്. ₹2,000-ൽ താഴെയുള്ള ചെറിയ അറ്റകുറ്റപ്പണികൾ വാടകക്കാരൻ്റെ ഉത്തരവാദിത്തമാണ്.",
        clauses: [
          {
            title: "സെക്യൂരിറ്റി ഡെപ്പോസിറ്റ് വ്യവസ്ഥ",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "നിങ്ങൾ ₹1,00,000 മുൻകൂറായി നൽകണം. മുറി ഒഴിയുമ്പോൾ കേടുപാടുകൾ തീർക്കുന്ന തുക കഴിച്ച് ബാക്കി പലിശയില്ലാത്ത തുക ഉടമ ഇത് മടക്കി നൽകേണ്ടതുണ്ട്."
          },
          {
            title: "വാടക വർദ്ധനവ്",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "11 മാസത്തിന് ശേഷം കരാർ പുതുക്കുകയാണെങ്കിൽ വാടക 10% വർദ്ധിക്കും (₹25,000 എന്നത് ₹27,500 ആകും)."
          },
          {
            title: "അറ്റകുറ്റപ്പണികൾ",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "₹2,000-ൽ താഴെയുള്ള ചെറിയ റിപ്പയർ ജോലികൾ (ലൈറ്റ് ബൾബ്, ടാപ്പ് കേടുപാടുകൾ) നിങ്ങൾ ചെയ്യണം. വലിയ അറ്റകുറ്റപ്പണികൾക്ക് ഉടമ പണം നൽകും."
          }
        ]
      },
      mr: {
        summary: "घरमालक राजेश शर्मा आणि भाडेकरू अमित वर्मा यांच्यात अपार्टमेंट 4B, ग्रीन मेडोजसाठी 11 महिन्यांचा भाडेकरार. भाडे दरमहा ₹२५,००० आहे, सुरक्षा ठेव ₹१,००,००० आहे आणि नूतनीकरणावर १०% भाडे वाढ आहे. ₹२,००० पेक्षा कमी दुरुस्ती भाडेकरूची जिम्मेदारी आहे.",
        clauses: [
          {
            title: "सुरक्षा ठेव करार",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "तुम्हाला ₹१,००,००० सुरक्षा ठेव (डिपॉझिट) द्यावी लागेल. घर सोडताना घरमालक नुकसानीचे पैसे वजा करून हे व्याजमुक्त डिपॉझिट परत करेल."
          },
          {
            title: "भाडे वाढ",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "११ महिन्यांनंतर कराराचे नूतनीकरण केल्यास भाडे १०% वाढेल (₹२५,००० वरून ₹२७,५०० होईल)."
          },
          {
            title: "देखभाल आणि दुरुस्ती",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "₹२,००० पेक्षा कमी खर्चाची लहान कामे (उदा. बल्ब किंवा नळ दुरुस्ती) तुम्हाला करावी लागतील. मोठे काम घरमालक करेल."
          }
        ]
      },
      bn: {
        summary: "বাড়িওয়ালা Rajesh Sharma এবং ভাড়াটিয়া Amit Verma-এর মধ্যে অ্যাপার্টমেন্ট 4B, Green Meadows-এর জন্য ১১ মাসের ভাড়ার চুক্তি। ভাড়া প্রতি মাসে ₹২৫,০০০, সিকিউরিটি ডিপোজিট ₹১,০০,০০০ এবং চুক্তি নবায়নের সময় ১০% ভাড়া বৃদ্ধি পাবে। ₹২,০০০ এর কম মূল্যের ছোটখাটো মেরামত ভাড়াটিয়ার দায়িত্ব।",
        clauses: [
          {
            title: "সিকিউরিটি ডিপোজিট ধারা",
            original: "The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.",
            simplified: "আপনাকে ₹১,০০,০০০ সিকিউরিটি ডিপোজিট দিতে হবে। আপনি যখন বাড়ি খালি করবেন, বাড়িওয়ালা কোনো ক্ষতি হলে তার খরচ কেটে নিয়ে বাকি সুদহীন টাকা ফেরত দেবেন।"
          },
          {
            title: "ভাড়া বৃদ্ধি",
            original: "Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.",
            simplified: "১১ মাস পর যদি চুক্তিটি নবায়ন করা হয়, তবে ভাড়া ১০% বৃদ্ধি পাবে (ভাড়া ₹২৫,০০০ থেকে বেড়ে ₹২৭,৫০০ হবে)।"
          },
          {
            title: "রক্ষণাবেক্ষণ ও মেরামত",
            original: "The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.",
            simplified: "₹২,০০০ এর নিচে ছোটখাটো মেরামতের খরচ (যেমন লাইট বা কলের কাজ) আপনাকে বহন করতে হবে। বড় ধরনের কাজ বাড়িওয়ালা করবেন।"
          }
        ]
      }
    }
  },
  {
    id: 'employment_agreement',
    name: 'Employment & Non-Disclosure Agreement',
    filename: 'Employment_and_NDA_Agreement.docx',
    size: '188 KB',
    data: {
      en: {
        summary: "An employment offer and non-disclosure agreement between TechNova Solutions and employee Sneha Sen for the position of Senior Frontend Engineer. The agreement includes a 3-month probation period, standard Intellectual Property (IP) assignment (the company owns all work created), and a strict 1-year non-compete clause post-termination.",
        clauses: [
          {
            title: "Probationary Employment",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "Your first 3 months are a trial. During this time, either you or the company can end the job by giving just a 7-day written notice."
          },
          {
            title: "Intellectual Property Rights",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "Everything you write, design, or build while working here or using company laptops belongs fully and immediately to the company. You cannot take or use it after leaving."
          },
          {
            title: "Non-Competition Restraints",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "For 1 year after leaving, you cannot take a job with or consult for any direct competitor of the company in the same business sector."
          }
        ]
      },
      te: {
        summary: "టెక్‌నోవా సొల్యూషన్స్ మరియు ఉద్యోగి స్నేహ సేన్ మధ్య సీనియర్ ఫ్రంటెండ్ ఇంజనీర్ ఉద్యోగం కోసం కుదిరిన ఒప్పందం. ఇందులో 3 నెలల ప్రొబేషన్ కాలం, మేధో సంపత్తి హక్కులు (IP) పూర్తిగా సంస్థకే చెందడం మరియు ఉద్యోగం ముగిసిన తర్వాత 1 సంవత్సరం పాటు పోటీ సంస్థల్లో చేరకూడదనే కఠినమైన నిబంధనలు ఉన్నాయి.",
        clauses: [
          {
            title: "ప్రొబేషన్ వ్యవధి నిబంధన",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "మీ మొదటి 3 నెలలు ఒక ట్రయల్ పీరియడ్. ఈ సమయంలో, మీరు లేదా కంపెనీ ఎవరైనా కేవలం 7 రోజుల నోటీసు ఇచ్చి ఉద్యోగాన్ని ముగించవచ్చు."
          },
          {
            title: "మేధో సంపత్తి హక్కులు (IP)",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "ఇక్కడ పని చేస్తున్నప్పుడు లేదా కంపెనీ వనరులను ఉపయోగించి మీరు తయారు చేసే కోడ్, డిజైన్లు లేదా ఉత్పత్తులు పూర్తిగా కంపెనీకే చెందుతాయి. మీరు వాటిని సొంతానికి వాడుకోలేరు."
          },
          {
            title: "పోటీ సంస్థలలో చేరకూడదనే నిబంధన",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "ఈ కంపెనీ నుండి బయటకు వచ్చిన తర్వాత 1 సంవత్సరం వరకు, మీరు దీనికి పోటీగా ఉన్న ఇతర సంస్థలలో ఉద్యోగంలో చేరకూడదు."
          }
        ]
      },
      hi: {
        summary: "टेकनोवा सॉल्यूशंस और कर्मचारी स्नेहा सेन के बीच सीनियर फ्रंटएंड इंजीनियर के पद के लिए रोजगार प्रस्ताव और गोपनीयता समझौता। समझौते में 3 महीने की परिवीक्षा (प्रोबेशन) अवधि, बौद्धिक संपदा (IP) सौंपना (कंपनी द्वारा निर्मित सभी काम का मालिकाना हक) और नौकरी छोड़ने के बाद 1 साल की गैर-प्रतिस्पर्धा (नॉन-कंपीट) शर्त शामिल है।",
        clauses: [
          {
            title: "परिवीक्षा (प्रोबेशन) अवधि क्लॉज",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "आपके पहले 3 महीने ट्रायल के रूप में होंगे। इस अवधि के दौरान, आप या कंपनी दोनों में से कोई भी केवल 7 दिन का लिखित नोटिस देकर नौकरी समाप्त कर सकता है।"
          },
          {
            title: "बौद्धिक संपदा अधिकार (IP Rights)",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "काम के दौरान या कंपनी के उपकरणों का उपयोग करके आपके द्वारा बनाया गया सारा काम (कोड, डिजाइन आदि) पूरी तरह से कंपनी का होगा। नौकरी छोड़ने के बाद आप इसे अपने साथ नहीं ले जा सकते।"
          },
          {
            title: "गैर-प्रतिस्पर्धा प्रतिबंध (नॉन-कंपीट)",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "नौकरी छोड़ने के बाद 1 साल तक आप कंपनी की किसी भी सीधी प्रतिस्पर्धी (कंपटीटर) कंपनी में काम नहीं कर सकते।"
          }
        ]
      },
      ta: {
        summary: "டெக்நோவா சொல்யூஷன்ஸ் மற்றும் பணியாளர் சினேகா சென் இடையே சீனியர் ஃப்ரண்ட்எண்ட் இன்ஜினியர் பணிக்கான வேலைவாய்ப்பு மற்றும் இரகசிய காப்பு ஒப்பந்தம். இதில் 3 மாத தகுதிகாண் காலம் (ப்ரோபேஷன்), அறிவுசார் சொத்துரிமை நிறுவனம் வசம் இருத்தல் மற்றும் வேலையை விட்டு விலகிய பின் 1 வருடத்திற்கு போட்டியாளர் நிறுவனங்களில் சேரக்கூடாது என்ற கடுமையான நிபந்தனை உள்ளது.",
        clauses: [
          {
            title: "தகுதிகாண் வேலைவாய்ப்பு (ப்ரோபேஷன்)",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "உங்களது முதல் 3 மாதங்கள் ஒரு சோதனைக் காலம். இந்த காலகட்டத்தில், நீங்களோ அல்லது நிறுவனமோ 7 நாட்கள் முன்னறிவிப்பு கொடுத்து பணியை நிறுத்திக் கொள்ளலாம்."
          },
          {
            title: "அறிவுசார் சொத்துரிமை (IP Rights)",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "பணியின் போதோ அல்லது நிறுவனத்தின் கணினி போன்றவற்றை பயன்படுத்தியோ நீங்கள் உருவாக்கும் அனைத்தும் நிறுவனத்திற்கே சொந்தம். நீங்கள் வெளியேறும்போது அவற்றை எடுத்துச் செல்ல முடியாது."
          },
          {
            title: "போட்டி நிறுவனங்களில் சேரக்கூடாது என்ற கட்டுப்பாடு",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "வேலையை விட்டு விலகிய பிறகு 1 வருடத்திற்கு இதன் நேரடி போட்டியாளர் நிறுவனங்களில் நீங்கள் பணியில் சேரவோ அல்லது ஆலோசனை வழங்கவோ கூடாது."
          }
        ]
      },
      kn: {
        summary: "ಟೆಕ್‌ನೋವಾ ಸೊಲ್ಯೂಷನ್ಸ್ ಮತ್ತು ಉದ್ಯೋಗಿ ಸ್ನೇಹಾ ಸೇನ್ ನಡುವೆ ಸೀನಿಯರ್ ಫ್ರಂಟ್‌ಎಂಡ್ ಇಂಜಿನಿಯರ್ ಹುದ್ದೆಗಾಗಿ ಉದ್ಯೋಗ ಒಪ್ಪಂದ. ಇದರಲ್ಲಿ 3 ತಿಂಗಳ ಪ್ರೊಬೇಷನ್ ಅವಧಿ, ಬೌದ್ಧಿಕ ಆಸ್ತಿ ಹಕ್ಕುಗಳ (IP) ಸಂಪೂರ್ಣ ಹಸ್ತಾಂತರ ಮತ್ತು ಕೆಲಸ ಬಿಟ್ಟ ನಂತರ 1 ವರ್ಷದವರೆಗೆ ಯಾವುದೇ ಸ್ಪರ್ಧಾತ್ಮಕ ಸಂಸ್ಥೆಯಲ್ಲಿ ಕೆಲಸ ಮಾಡದಂತೆ ಕಠಿಣ ಷರತ್ತು ಇದೆ.",
        clauses: [
          {
            title: "ಪ್ರೊಬೇಷನರಿ ಉದ್ಯೋಗ",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "ನಿಮ್ಮ ಮೊದಲ 3 ತಿಂಗಳುಗಳು ಪ್ರೊಬೇಷನರಿ (ಪ್ರಾಯೋಗಿಕ) ಅವಧಿಯಾಗಿರುತ್ತದೆ. ಈ ಅವಧಿಯಲ್ಲಿ ಯಾವುದೇ ಪಕ್ಷವು ಕೇವಲ 7 ದಿನಗಳ ಲಿಖಿತ ನೋಟಿಸ್ ನೀಡಿ ಕೆಲಸ ಕೊನೆಗೊಳಿಸಬಹುದು."
          },
          {
            title: "ಬೌದ್ಧಿಕ ಆಸ್ತಿ ಹಕ್ಕುಗಳು (IP)",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "ಕೆಲಸದ ಸಮಯದಲ್ಲಿ ಅಥವಾ ಕಂಪನಿಯ ಸಂಪನ್ಮೂಲ ಬಳಸಿ ನೀವು ಅಭಿವೃದ್ಧಿಪಡಿಸುವ ಯಾವುದೇ ಕೋಡ್ ಅಥವಾ ಡಿಸೈನ್ ಸಂಪೂರ್ಣವಾಗಿ ಕಂಪನಿಗೆ ಸೇರುತ್ತದೆ. ಕೆಲಸ ಬಿಟ್ಟ ನಂತರ ಬಳಸುವಂತಿಲ್ಲ."
          },
          {
            title: "ಸ್ಪರ್ಧಾತ್ಮಕ ಸಂಸ್ಥೆಗಳಲ್ಲಿ ಕೆಲಸ ಮಾಡದಿರುವ ನಿಯಮ",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "ಕೆಲಸ ಬಿಟ್ಟ ನಂತರ 1 ವರ್ಷದವರೆಗೆ ನೀವು ಈ ಕಂಪನಿಯ ಯಾವುದೇ ನೇರ ಸ್ಪರ್ಧಿ ಕಂಪನಿಗಳಲ್ಲಿ ಕೆಲಸಕ್ಕೆ ಸೇರುವಂತಿಲ್ಲ."
          }
        ]
      },
      ml: {
        summary: "ടെക്നോവ സൊല്യൂഷൻസും സ്നേഹ സെൻ എന്ന ഉദ്യോഗസ്ഥയും തമ്മിലുള്ള സീനിയർ ഫ്രണ്ട് എൻഡ് എഞ്ചിനീയർ തസ്തികയിലേക്കുള്ള ജോബ് കരാറും നോൺ-ഡിസ്ക്ലോഷർ കരാറും. ഇതിൽ 3 മാസത്തെ പ്രൊബേഷൻ കാലയളവ്, ബൗദ്ധിക സ്വത്തവകാശം (IP) കൈമാറ്റം, ജോലി അവസാനിപ്പിച്ച ശേഷം 1 വർഷത്തേക്ക് സമാന കമ്പനികളിൽ ജോലി ചെയ്യുന്നതിനുള്ള വിലക്ക് എന്നിവ ഉൾപ്പെടുന്നു.",
        clauses: [
          {
            title: "പ്രൊബേഷൻ കാലയളവ്",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "നിങ്ങളുടെ ആദ്യ 3 മാസം ട്രയൽ പീരിയഡ് ആയിരിക്കും. ഈ സമയത്ത് നിങ്ങൾക്കോ കമ്പനിക്കോ വെറും 7 ദിവസത്തെ രേഖാമൂലമുള്ള നോട്ടീസ് നൽകി ജോലി അവസാനിപ്പിക്കാം."
          },
          {
            title: "ബൗദ്ധിക സ്വത്തവകാശം (IP)",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "ജോലി സമയത്തോ കമ്പനി ലാപ്ടോപ്പുകൾ വഴിയോ നിങ്ങൾ നിർമ്മിക്കുന്ന കോഡ്, ഡിസൈനുകൾ എന്നിവയെല്ലാം കമ്പനിയുടെ സ്വന്തമാണ്. ജോലി വിട്ട ശേഷം ഇവ ഉപയോഗിക്കാൻ സാധിക്കില്ല."
          },
          {
            title: "മത്സര വിലക്ക് (നോൺ-കംപീറ്റ്)",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "ജോലിയിൽ നിന്ന് പിരിഞ്ഞുപോയ ശേഷം 1 വർഷത്തേക്ക് കമ്പനിയുടെ പ്രധാന എതിരാളികളായ കമ്പനികളിൽ നിങ്ങൾ ജോലി ചെയ്യാൻ പാടുള്ളതല്ല."
          }
        ]
      },
      mr: {
        summary: "टेकनोव्हा सोल्युशन्स आणि स्नेहा सेन यांच्यात सीनियर फ्रंटएंड इंजिनीअर पदासाठी रोजगार आणि गोपनीयता करार. या करारात ३ महिन्यांचा प्रोबेशन काळ, बौद्धिक संपदा (IP) मालकी आणि नोकरी सोडल्यानंतर १ वर्षापर्यंत थेट प्रतिस्पर्धी कंपन्यांमध्ये न जाण्याची अट समाविष्ट आहे.",
        clauses: [
          {
            title: "परिवीक्षा (प्रोबेशन) कालावधी",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "तुमचे पहिले ३ महिने प्रोबेशन (चाचणी) कालावधी असेल. या काळात तुम्ही किंवा कंपनी, कोणीही ७ दिवसांची लेखी नोटीस देऊन नोकरी संपवू शकते."
          },
          {
            title: "बौद्धिक संपदा अधिकार",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "कामाच्या दरम्यान किंवा कंपनीचे साहित्य वापरून तुम्ही जे काही बनवाल (उदा. कोड, डिझाइन), त्याची मालकी पूर्णपणे कंपनीकडे राहील. तुम्ही ते स्वतः वापरू शकत नाही."
          },
          {
            title: "गैर-प्रतिस्पर्धा करार",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "नोकरी सोडल्यानंतर १ वर्षापर्यंत तुम्ही कंपनीच्या कोणत्याही थेट प्रतिस्पर्धी कंपनीत नोकरी करू शकत नाही."
          }
        ]
      },
      bn: {
        summary: "টেকনোভা সলিউশন এবং কর্মী স্নেহা সেনের মধ্যে সিনিয়র ফ্রন্টএন্ড ইঞ্জিনিয়ারের পদের জন্য কর্মসংস্থান এবং গোপনীয়তার চুক্তি। চুক্তিতে ৩ মাসের শিক্ষানবিস (প্রোবেশন) সময়কাল, বুদ্ধিবৃত্তিক সম্পত্তি (IP) হস্তান্তর এবং চাকরি ছাড়ার পর ১ বছর কোনো প্রতিযোগী সংস্থায় যোগ না দেওয়ার শর্ত রয়েছে।",
        clauses: [
          {
            title: "শিক্ষানবিস (প্রোবেশন) সময়কাল",
            original: "The Employee's initial employment period shall be probationary for a duration of three (3) calendar months from the commencement date, during which period either party may terminate the employment contract by providing seven (7) days written notice.",
            simplified: "আপনার প্রথম ৩ মাস ট্রায়াল পিরিয়ড। এই সময়ের মধ্যে, আপনি বা কোম্পানি যে কেউ মাত্র ৭ দিনের নোটিশে চাকরিটি বন্ধ করতে পারেন।"
          },
          {
            title: "বুদ্ধিবৃত্তিক সম্পত্তি অধিকার (IP Rights)",
            original: "All proprietary concepts, code bases, methodologies, patents, and work products developed, authored, or conceived by the Employee during working hours or using Company resources shall vest solely, immediately, and exclusively with the Employer.",
            simplified: "কাজের সময় বা কোম্পানির জিনিসপত্র ব্যবহার করে আপনার তৈরি করা সমস্ত কোড, ডিজাইন বা প্রোডাক্ট সম্পূর্ণরূপে কোম্পানির হবে। চাকরি ছাড়ার পর আপনি তা ব্যবহার করতে পারবেন না।"
          },
          {
            title: "প্রতিযোগিতা না করার শর্ত",
            original: "For a period of twelve (12) consecutive months immediately following termination of employment for any reason, the Employee shall not directly or indirectly engage with, or accept employment from, entities deemed direct competitors.",
            simplified: "চাকরি ছাড়ার পর ১ বছর পর্যন্ত আপনি এই কোম্পানির কোনো সরাসরি প্রতিযোগী সংস্থায় চাকরি নিতে পারবেন না।"
          }
        ]
      }
    }
  }
];

// Fallback translations if user drops their own custom document
export const CUSTOM_DOCUMENT_FALLBACK = {
  name: "Uploaded Legal Document",
  data: {
    en: {
      summary: "A customized legal document submitted for analysis. The document details mutual operational guidelines, compliance frameworks, and binding schedules between the contracting parties, carrying standard confidentiality and dispute resolution provisions.",
      clauses: [
        {
          title: "Confidentiality Obligations",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "You must keep all shared business secrets private. You cannot share them with anyone else unless you get written permission first."
        },
        {
          title: "Dispute Resolution and Jurisdiction",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "If you have a disagreement, you cannot go straight to court. You must resolve it through a neutral arbitrator in the city where the company's head office is located."
        }
      ]
    },
    te: {
      summary: "విశ్లేషణ కోసం సమర్పించబడిన అనుకూలీకరించిన చట్టపరమైన పత్రం. ఇది పార్టీల మధ్య పరస్పర కార్యాచరణ మార్గదర్శకాలు, నిబంధనలు మరియు వివాద పరిష్కార నిబంధనలను కలిగి ఉంటుంది.",
      clauses: [
        {
          title: "రహస్య సమాచార నిబంధన",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "పంచుకోబడిన వ్యాపార రహస్యాలను మీరు చాలా జాగ్రత్తగా దాచాలి. ముందుగా రాతపూర్వక అనుమతి లేకుండా వీటిని ఇతరులతో పంచుకోకూడదు."
        },
        {
          title: "వివాదాల పరిష్కారం మరియు అధికార పరిధి",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "ఏదైనా గొడవ జరిగితే నేరుగా కోర్టుకు వెళ్లలేరు. కంపెనీ హెడ్ ఆఫీస్ ఉన్న నగరంలో మధ్యవర్తిత్వం (ఆర్బిట్రేషన్) ద్వారా పరిష్కరించుకోవాలి."
        }
      ]
    },
    hi: {
      summary: "विश्लेषण के लिए प्रस्तुत किया गया कानूनी दस्तावेज। इस दस्तावेज में दोनों पक्षों के बीच आपसी परिचालन दिशानिर्देश, अनुपालन और विवाद समाधान प्रावधान शामिल हैं।",
      clauses: [
        {
          title: "गोपनीयता बाध्यता",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "साझा किए गए व्यावसायिक रहस्यों को आपको गुप्त रखना होगा। लिखित अनुमति के बिना आप इन्हें किसी तीसरे पक्ष के साथ साझा नहीं कर सकते।"
        },
        {
          title: "विवाद समाधान और क्षेत्राधिकार",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "विवाद होने पर आप सीधे कोर्ट नहीं जा सकते। इसे कंपनी मुख्यालय वाले शहर में मध्यस्थता (Arbitration) द्वारा सुलझाना होगा।"
        }
      ]
    },
    ta: {
      summary: "பகுப்பாய்விற்காக சமர்ப்பிக்கப்பட்ட தனிப்பயனாக்கப்பட்ட சட்ட ஆவணம். இதில் பரஸ்பர செயல்பாட்டு வழிகாட்டுதல்கள் மற்றும் தகராறு தீர்வு விதிகள் உள்ளன.",
      clauses: [
        {
          title: "இரகசிய காப்பு கடமைகள்",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "பகிர்ந்துகொள்ளப்பட்ட ரகசியங்களை நீங்கள் பாதுகாக்க வேண்டும். முறையான எழுத்துப்பூர்வ அனுமதியின்றி மற்றவர்களுடன் இதைப் பகிரக் கூடாது."
        },
        {
          title: "தகராறு தீர்வு மற்றும் அதிகார வரம்பு",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "ஏதேனும் சர்ச்சை ஏற்பட்டால் நேரடியாக நீதிமன்றம் செல்ல முடியாது. நிறுவன தலைமையகம் அமைந்துள்ள நகரில் நடுவர் மன்றம் (Arbitration) மூலமே தீர்க்க வேண்டும்."
        }
      ]
    },
    kn: {
      summary: "ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಸಲ್ಲಿಕೆಯಾದ ಕಸ್ಟಮ್ ಕಾನೂನು ದಾಖಲೆ. ಇದು ಪಕ್ಷಗಳ ನಡುವಿನ ಪರಸ್ಪರ ಕಾರ್ಯಾಚರಣೆಯ ಮಾರ್ಗಸೂಚಿಗಳು ಮತ್ತು ವಿವಾದ ಪರಿಹಾರ ನಿಯಮಗಳನ್ನು ಒಳಗೊಂಡಿದೆ.",
      clauses: [
        {
          title: "ಗೋಪ್ಯತೆಯ ಜವಾಬ್ದಾರಿಗಳು",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "ಹಂಚಿಕೊಳ್ಳಲಾದ ವ್ಯಾಪಾರ ರಹಸ್ಯಗಳನ್ನು ನೀವು ಗೋಪ್ಯವಾಗಿಡಬೇಕು. ಲಿಖಿತ ಅನುಮತಿ ಇಲ್ಲದೆ ಯಾರೊಂದಿಗೂ ಹಂಚಿಕೊಳ್ಳುವಂತಿಲ್ಲ."
        },
        {
          title: "ವಿವಾದ ಪರಿಹಾರ ಮತ್ತು ಅಧಿಕಾರ ವ್ಯಾಪ್ತಿ",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "ಯಾವುದೇ ವಿವಾದವಿದ್ದಲ್ಲಿ ನೇರವಾಗಿ ಕೋರ್ಟ್‌ಗೆ ಹೋಗುವಂತಿಲ್ಲ. ಕಂಪನಿಯ ಪ್ರಧಾನ ಕಚೇರಿ ಇರುವ ನಗರದಲ್ಲೇ ಸಂಧಾನಕಾರರ (ಆರ್ಬಿಟ್ರೇಷನ್) ಮೂಲಕ ಬಗೆಹರಿಸಿಕೊಳ್ಳಬೇಕು."
        }
      ]
    },
    ml: {
      summary: "വിശകലനത്തിനായി നൽകിയ കസ്റ്റം നിയമ രേഖ. ഇതിൽ ഇരു കക്ഷികളും തമ്മിലുള്ള പ്രവർത്തന മാർഗ്ഗനിർദ്ദേശങ്ങളും തർക്ക പരിഹാര വ്യവസ്ഥകളും അടങ്ങിയിരിക്കുന്നു.",
      clauses: [
        {
          title: "രഹസ്യാത്മകത പാലിക്കൽ",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "പങ്കുവെച്ച ബിസിനസ്സ് രഹസ്യങ്ങൾ നിങ്ങൾ സൂക്ഷിക്കണം. മുൻകൂട്ടി എഴുതി തയ്യാറാക്കിയ അനുമതിയില്ലാതെ ഇത് മറ്റുള്ളവരുമായി പങ്കുവെക്കരുത്."
        },
        {
          title: "തർക്ക പരിഹാരവും അധികാരപരിധിയും",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "തർക്കങ്ങൾ ഉണ്ടായാൽ കോടതിയിൽ പോകാൻ കഴിയില്ല. കമ്പനി ഹെഡ് ഓഫീസ് സ്ഥിതി ചെയ്യുന്ന നഗരത്തിൽ വെച്ച് മധ്യസ്ഥത (Arbitration) വഴി പരിഹരിക്കണം."
        }
      ]
    },
    mr: {
      summary: "विश्लेषणासाठी सादर केलेले वैयक्तिकृत कायदेशीर दस्तऐवज. यामध्ये परस्पर परिचालन मार्गदर्शक तत्त्वे आणि विवाद निवारण तरतुदी समाविष्ट आहेत.",
      clauses: [
        {
          title: "गोपनीयतेची बंधने",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "सामायिक केलेले व्यावसायिक गुपिते गुप्त ठेवणे बंधनकारक आहे. पूर्वपरवानगीशिवाय ते इतरांना सांगू नका."
        },
        {
          title: "विवाद निवारण आणि अधिकार क्षेत्र",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "वाद उद्भवल्यास तुम्ही थेट कोर्टात जाऊ शकत नाही. कंपनी मुख्यालय असलेल्या शहरात लवादामार्फत (Arbitration) तो सोडवावा लागेल."
        }
      ]
    },
    bn: {
      summary: "বিশ্লেষণের জন্য জমা দেওয়া কাস্টম আইনি নথি। এটিতে উভয় পক্ষের মধ্যে পারস্পরিক অপারেশনাল নির্দেশিকা এবং বিরোধ নিষ্পত্তির নিয়ম রয়েছে।",
      clauses: [
        {
          title: "গোপনীয়তার বাধ্যবাধকতা",
          original: "Each receiving party agrees to maintain strict confidentiality of all proprietary disclosures designated as sensitive, and shall not disclose same to third parties without prior written express consent from the disclosing party.",
          simplified: "শেয়ার করা গোপন তথ্য আপনাকে গোপন রাখতে হবে। আগে থেকে লিখিত অনুমতি ছাড়া আপনি তা কাউকে জানাতে পারবেন না।"
        },
        {
          title: "বিরোধ নিষ্পত্তি ও এখতিয়ার",
          original: "Any controversy, claim, or dispute arising out of or relating to this contract shall be settled by binding arbitration in accordance with local arbitration statutes, with the venue of proceedings located in the jurisdiction of the corporate headquarters.",
          simplified: "কোনো বিরোধ দেখা দিলে সরাসরি আদালতে যাওয়া যাবে না। কোম্পানির প্রধান কার্যালয় অবস্থিত শহরে সালিশির (Arbitration) মাধ্যমে তা সমাধান করতে হবে।"
        }
      ]
    }
  }
};
