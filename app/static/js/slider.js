function updateIntInput(val, name) {
    document.getElementById(name).value = val; 
}

function updateFloatInput(val, name) {
    document.getElementById(name).value = val/100; 
}

$('.field-tip').tooltip({
    disabled: true,
    close: function( event, ui ) { $(this).tooltip('disable'); }
});

$('.field-tip').on('click', function () {
    $(this).tooltip('enable').tooltip('open');
});

function updateLambda(lambda, gen_rate, flow_rate) {
    
    document.getElementById('conc').value =  gen_rate * lambda * 60 / flow_rate

}

function disableSlider(id, targetSlider, targetText) {

    var value = document.getElementById(id)

    if(value.checked) {
        document.getElementById(targetSlider).disabled = false;
        document.getElementById(targetText).disabled = false;
        document.getElementById(targetSlider).classList.add('slider-activated');
        document.getElementById(targetSlider).classList.remove('slider');
    }
    else {
        document.getElementById(targetSlider).disabled = true;
        document.getElementById(targetText).disabled = true;
        document.getElementById(targetSlider).classList.remove('slider-activated');
        document.getElementById(targetSlider).classList.add('slider');
    }
}

function disableCombo(id, targetCombo) {

    var value = document.getElementById(id)

    if(value.checked) {
        document.getElementById(targetCombo).disabled = false;
    }
    else {
        document.getElementById(targetCombo).disabled = true;
    }
}